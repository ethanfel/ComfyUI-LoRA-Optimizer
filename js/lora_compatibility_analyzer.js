import { app } from "/scripts/app.js";

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

app.registerExtension({
    name: "LoRAOptimizer.CompatibilityAnalyzer",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRACompatibilityAnalyzer") return;

        if (!node.properties) node.properties = {};
        if (!node.properties.createdGroupIds) node.properties.createdGroupIds = [];

        // Clean up child nodes when analyzer is deleted
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            const tracked = this.properties?.createdGroupIds || [];
            for (const id of tracked) {
                if (id != null) {
                    const child = app.graph.getNodeById(id);
                    if (child && (child.comfyClass === "LoRAStackDynamic" || child.comfyClass === "LoraLoader" || child.comfyClass === "LoraLoaderModelOnly")) {
                        app.graph.remove(child);
                    }
                }
            }
            if (origOnRemoved) origOnRemoved.call(this);
        };

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.call(this, message);

            const groups = message?.groups;
            if (!groups) return;

            const hasClip = message.has_clip?.[0] !== false;
            const tracked = node.properties.createdGroupIds || [];
            const newTracked = [];

            let mergeGroupNum = 0;
            const loaderClass = hasClip ? "LoraLoader" : "LoraLoaderModelOnly";

            // Update or create nodes for each group
            for (let i = 0; i < groups.length; i++) {
                const group = groups[i];
                const isLoader = group.node_type === "loader";
                const expectedClass = isLoader ? loaderClass : "LoRAStackDynamic";

                let childNode = null;

                // Try to reuse existing tracked node
                if (i < tracked.length && tracked[i] != null) {
                    childNode = app.graph.getNodeById(tracked[i]);
                    // Remove if wrong type
                    if (childNode && childNode.comfyClass !== expectedClass) {
                        app.graph.remove(childNode);
                        childNode = null;
                    }
                }

                if (childNode && childNode.comfyClass === expectedClass) {
                    // Update existing node
                    if (isLoader) {
                        populateLoaderNode(childNode, group);
                        childNode.title = "Solo (Analyzer)";
                    } else {
                        mergeGroupNum++;
                        populateStackNode(childNode, group, i);
                        childNode.title = `Group ${mergeGroupNum} (Analyzer)`;
                    }
                    newTracked.push(childNode.id);
                } else {
                    // Create new node
                    const created = LiteGraph.createNode(expectedClass);
                    if (!created) {
                        console.warn(`[CompatibilityAnalyzer] ${expectedClass} node type not found`);
                        continue;
                    }
                    app.graph.add(created);
                    created.pos = [
                        node.pos[0] + node.size[0] + 50,
                        node.pos[1] + i * 250,
                    ];

                    if (isLoader) {
                        created.title = "Solo (Analyzer)";
                        populateLoaderNode(created, group);
                    } else {
                        mergeGroupNum++;
                        created.title = `Group ${mergeGroupNum} (Analyzer)`;
                        // Delay population to let widgets initialize (must exceed
                        // the 100ms initial-visibility timeout in lora_stack_dynamic.js)
                        setTimeout(() => populateStackNode(created, group, i), 200);
                    }
                    newTracked.push(created.id);
                }
            }

            // Remove orphaned nodes (groups shrank)
            for (let i = groups.length; i < tracked.length; i++) {
                if (tracked[i] != null) {
                    const orphan = app.graph.getNodeById(tracked[i]);
                    if (orphan && (orphan.comfyClass === "LoRAStackDynamic" || orphan.comfyClass === "LoraLoader" || orphan.comfyClass === "LoraLoaderModelOnly")) {
                        app.graph.remove(orphan);
                    }
                }
            }

            node.properties.createdGroupIds = newTracked;
            app.canvas?.setDirty?.(true, true);
        };
    },
});

function populateLoaderNode(loaderNode, group) {
    const nameWidget = findWidget(loaderNode, "lora_name");
    if (nameWidget) nameWidget.value = group.lora_name;
    const modelStr = findWidget(loaderNode, "strength_model");
    if (modelStr) modelStr.value = group.strength;
    const clipStr = findWidget(loaderNode, "strength_clip");
    if (clipStr) clipStr.value = group.strength;
}

function populateStackNode(stackNode, group, groupIndex) {
    const modeWidget = findWidget(stackNode, "input_mode");
    if (modeWidget) modeWidget.value = "text";

    const countWidget = findWidget(stackNode, "lora_count");
    if (countWidget) countWidget.value = group.loras.length;

    // Populate active slots and clear any stale surplus slots
    const MAX_SLOTS = 10;
    for (let i = 0; i < MAX_SLOTS; i++) {
        const textWidget = findWidget(stackNode, `lora_name_text_${i + 1}`);
        const strengthWidget = findWidget(stackNode, `strength_${i + 1}`);
        if (i < group.loras.length) {
            if (textWidget) textWidget.value = group.loras[i].name;
            if (strengthWidget) strengthWidget.value = group.loras[i].strength;
        } else {
            if (textWidget) textWidget.value = "None";
            if (strengthWidget) strengthWidget.value = 1.0;
        }
    }
}
