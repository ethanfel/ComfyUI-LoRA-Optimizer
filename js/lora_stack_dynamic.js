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

function updateVisibility(node) {
    const modeWidget = findWidget(node, "mode");
    const countWidget = findWidget(node, "lora_count");
    if (!modeWidget || !countWidget) return;

    const isSimple = modeWidget.value === "simple";
    const count = countWidget.value;
    const MAX = 10;

    for (let i = 1; i <= MAX; i++) {
        const visible = i <= count;

        toggleWidget(node, findWidget(node, `lora_name_${i}`), visible);
        toggleWidget(node, findWidget(node, `strength_${i}`), visible && isSimple);
        toggleWidget(node, findWidget(node, `model_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `clip_strength_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `conflict_mode_${i}`), visible);
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
    app.canvas?.setDirty?.(true, true);
}

app.registerExtension({
    name: "LoRAOptimizer.LoRAStackDynamic",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAStackDynamic") return;

        // Intercept mode and lora_count changes to update visibility
        for (const w of node.widgets || []) {
            if (w.name !== "mode" && w.name !== "lora_count") continue;

            let widgetValue = w.value;
            const originalDescriptor =
                Object.getOwnPropertyDescriptor(w, "value") ||
                Object.getOwnPropertyDescriptor(
                    Object.getPrototypeOf(w),
                    "value"
                );

            Object.defineProperty(w, "value", {
                get() {
                    return originalDescriptor?.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;
                },
                set(newVal) {
                    if (originalDescriptor?.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }
                    updateVisibility(node);
                },
            });
        }

        // Initial visibility update — delay to ensure widgets are fully initialized
        setTimeout(() => updateVisibility(node), 100);
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
