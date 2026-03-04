import { app } from "/scripts/app.js";

const HIDDEN_TAG = "loraopt_hidden";
const origProps = {};

function toggleWidget(node, widget, show, suffix = "") {
    if (!widget) return;

    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
        };
    }

    widget.hidden = !show;
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show
        ? origProps[widget.name].origComputeSize
        : () => [0, -4];

    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            toggleWidget(node, w, show, ":" + widget.name);
        }
    }
}

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function interceptWidgetValue(widget, onChange) {
    let widgetValue = widget.value;
    const desc =
        Object.getOwnPropertyDescriptor(widget, "value") ||
        Object.getOwnPropertyDescriptor(
            Object.getPrototypeOf(widget),
            "value"
        );

    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() {
            return desc?.get ? desc.get.call(widget) : widgetValue;
        },
        set(newVal) {
            if (desc?.set) {
                desc.set.call(widget, newVal);
            } else {
                widgetValue = newVal;
            }
            onChange(newVal);
        },
    });
}

function updateVisibility(node) {
    const modeWidget = findWidget(node, "mode");
    const inputModeWidget = findWidget(node, "input_mode");
    const countWidget = findWidget(node, "lora_count");
    if (!modeWidget || !inputModeWidget || !countWidget) return;

    const isSimple = modeWidget.value === "simple";
    const isText = inputModeWidget.value === "text";
    const count = countWidget.value;
    const MAX = 10;

    for (let i = 1; i <= MAX; i++) {
        const visible = i <= count;

        toggleWidget(node, findWidget(node, `lora_name_${i}`), visible && !isText);
        toggleWidget(node, findWidget(node, `lora_name_text_${i}`), visible && isText);
        toggleWidget(node, findWidget(node, `strength_${i}`), visible && isSimple);
        toggleWidget(node, findWidget(node, `model_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `clip_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `conflict_mode_${i}`), visible);
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
    app.canvas?.setDirty?.(true, true);
}

// --- Node Registration ---

app.registerExtension({
    name: "LoRAOptimizer.LoRAStackDynamic",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAStackDynamic") return;

        // Intercept mode, input_mode, and lora_count changes to update visibility
        for (const w of node.widgets || []) {
            if (w.name !== "mode" && w.name !== "input_mode" && w.name !== "lora_count") continue;
            interceptWidgetValue(w, () => updateVisibility(node));
        }

        // Initial visibility update — delay to ensure widgets are fully initialized
        setTimeout(() => {
            updateVisibility(node);
        }, 100);
    },
});

app.registerExtension({
    name: "LoRAOptimizer.LoRAConflictEditor",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAConflictEditor") return;
        // All 10 conflict_mode slots are always visible.
        // Unused slots (beyond the stack size) default to "auto" and are ignored.
    },
});
