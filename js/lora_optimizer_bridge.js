import { app } from "/scripts/app.js";

/**
 * AutoTuner ↔ Optimizer Bridge
 *
 * 1. Widget sync: when the Optimizer runs with settings_source="from_autotuner",
 *    applied_settings in the UI message updates the Optimizer's knobs.
 *
 * 2. Opposing switches: when tuner_data is connected between an AutoTuner and
 *    an Optimizer, their mode switches stay in sync as opposites:
 *      Optimizer "from_autotuner" ↔ AutoTuner "merge"
 *      Optimizer "manual"         ↔ AutoTuner "tuning_only"
 *
 * 3. Conditional visibility: settings_source (Optimizer) and output_mode
 *    (AutoTuner) are hidden until tuner_data is connected.
 */

const HIDDEN_TAG = "loraopt_bridge_hidden";
const _origProps = {};

function toggleWidget(node, widget, show) {
    if (!widget) return;
    const key = node.id + ":" + widget.name;
    if (!_origProps[key]) {
        _origProps[key] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
        };
    }
    widget.hidden = !show;
    widget.type = show ? _origProps[key].origType : HIDDEN_TAG;
    widget.computeSize = show ? _origProps[key].origComputeSize : () => [0, -4];
}

const BRIDGE_WIDGETS = [
    "optimization_mode",
    "merge_quality",
    "sparsification",
    "sparsification_density",
    "dare_dampening",
    "auto_strength",
    "auto_strength_floor",
];

const OPTIMIZER_TO_AUTOTUNER = {
    "from_autotuner": "merge",
    "manual": "tuning_only",
};
const AUTOTUNER_TO_OPTIMIZER = {
    "merge": "from_autotuner",
    "tuning_only": "manual",
};

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
}

function findConnectedAutoTuner(optimizerNode) {
    if (!optimizerNode.inputs) return null;
    for (let index = 0; index < optimizerNode.inputs.length; index++) {
        const input = optimizerNode.inputs[index];
        if (input.name === "tuner_data" && input.link != null) {
            const linkInfo = app.graph.links[input.link];
            if (linkInfo) {
                const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
                if (sourceNode && sourceNode.comfyClass === "LoRAAutoTuner") {
                    return sourceNode;
                }
            }
        }
    }
    return null;
}

function findConnectedOptimizer(autotunerNode) {
    if (!autotunerNode.outputs) return null;
    for (let index = 0; index < autotunerNode.outputs.length; index++) {
        const output = autotunerNode.outputs[index];
        if (output.name === "tuner_data" && output.links) {
            for (const linkId of output.links) {
                const linkInfo = app.graph.links[linkId];
                if (linkInfo) {
                    const targetNode = app.graph.getNodeById(linkInfo.target_id);
                    if (targetNode && targetNode.comfyClass === "LoRAOptimizer") {
                        return targetNode;
                    }
                }
            }
        }
    }
    return null;
}

function isTunerDataInputConnected(optimizerNode) {
    if (!optimizerNode.inputs) return false;
    for (const input of optimizerNode.inputs) {
        if (input.name === "tuner_data" && input.link != null) return true;
    }
    return false;
}

function isTunerDataOutputConnected(autotunerNode) {
    if (!autotunerNode.outputs) return false;
    for (const output of autotunerNode.outputs) {
        if (output.name === "tuner_data" && output.links && output.links.length > 0) return true;
    }
    return false;
}

function resizeNode(node) {
    const computed = node.computeSize();
    node.setSize([Math.max(node.size[0], computed[0]), computed[1]]);
    app.canvas?.setDirty?.(true, true);
}

function updateOptimizerVisibility(node) {
    const widget = findWidget(node, "settings_source");
    if (!widget) return;
    const connected = isTunerDataInputConnected(node);
    toggleWidget(node, widget, connected);
    if (!connected) widget.value = "manual";
    resizeNode(node);
}

function updateAutoTunerVisibility(node) {
    const widget = findWidget(node, "output_mode");
    if (!widget) return;
    const connected = isTunerDataOutputConnected(node);
    toggleWidget(node, widget, connected);
    if (!connected) widget.value = "merge";
    resizeNode(node);
}

let _syncing = false;

function syncOppositeSwitch(sourceWidget, sourceNode, targetFinder, mapping) {
    if (_syncing) return;
    const targetNode = targetFinder(sourceNode);
    if (!targetNode) return;

    const targetWidgetName = sourceWidget.name === "settings_source" ? "output_mode" : "settings_source";
    const targetWidget = findWidget(targetNode, targetWidgetName);
    if (!targetWidget) return;

    const expectedValue = mapping[sourceWidget.value];
    if (expectedValue && targetWidget.value !== expectedValue) {
        _syncing = true;
        try {
            targetWidget.value = expectedValue;
        } finally {
            _syncing = false;
        }
        app.canvas?.setDirty?.(true, true);
    }
}

app.registerExtension({
    name: "LoRAOptimizer.AutoTunerBridge",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAOptimizer") return;

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            if (origOnExecuted) {
                origOnExecuted.call(this, message);
            }

            if (!message || !message.applied_settings || !message.applied_settings[0]) {
                return;
            }

            let config;
            try {
                config = JSON.parse(message.applied_settings[0]);
            } catch {
                return;
            }

            for (const name of BRIDGE_WIDGETS) {
                const widget = findWidget(this, name);
                if (widget && config[name] !== undefined) {
                    widget.value = config[name];
                }
            }

            app.canvas?.setDirty?.(true, true);
        };

        const settingsWidget = findWidget(node, "settings_source");
        if (settingsWidget) {
            const origCallback = settingsWidget.callback;
            settingsWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                syncOppositeSwitch(settingsWidget, node, findConnectedAutoTuner, OPTIMIZER_TO_AUTOTUNER);
            };
        }

        setTimeout(() => updateOptimizerVisibility(node), 0);
        const origOnConnChange = node.onConnectionsChange;
        node.onConnectionsChange = function (side, slot, connected, linkInfo, ioSlot) {
            if (origOnConnChange) origOnConnChange.apply(this, arguments);
            if (side === 1) updateOptimizerVisibility(this);
        };
    },
});

app.registerExtension({
    name: "LoRAOptimizer.AutoTunerOutputMode",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAAutoTuner") return;

        const outputModeWidget = findWidget(node, "output_mode");
        if (outputModeWidget) {
            const origCallback = outputModeWidget.callback;
            outputModeWidget.callback = function (value) {
                if (origCallback) origCallback.call(this, value);
                syncOppositeSwitch(outputModeWidget, node, findConnectedOptimizer, AUTOTUNER_TO_OPTIMIZER);
            };
        }

        setTimeout(() => updateAutoTunerVisibility(node), 0);
        const origOnConnChange = node.onConnectionsChange;
        node.onConnectionsChange = function (side, slot, connected, linkInfo, ioSlot) {
            if (origOnConnChange) origOnConnChange.apply(this, arguments);
            if (side === 2) updateAutoTunerVisibility(this);
        };
    },
});
