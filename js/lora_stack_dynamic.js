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

function syncLoraValues(node, toText) {
    const MAX = 10;
    for (let i = 1; i <= MAX; i++) {
        const combo = findWidget(node, `lora_name_${i}`);
        const text = findWidget(node, `lora_name_text_${i}`);
        if (!combo || !text) continue;

        if (toText) {
            // dropdown -> text: copy filename stem (without path/extension)
            const val = combo.value || "None";
            if (val === "None") {
                text.value = "None";
            } else {
                const filename = val.split("/").pop() || val;
                text.value = filename.replace(/\.[^.]+$/, "");
            }
        } else {
            // text -> dropdown: try to match text against COMBO options
            const val = (text.value || "").trim();
            const options = combo.options?.values || [];
            const match = options.find((o) => o === val)
                || options.find((o) => o.toLowerCase().endsWith("/" + val.toLowerCase() + ".safetensors"))
                || options.find((o) => {
                    const stem = o.split("/").pop()?.replace(/\.[^.]+$/, "") || "";
                    return stem.toLowerCase() === val.toLowerCase();
                });
            if (match) {
                combo.value = match;
            }
        }
    }
}

function updateVisibility(node) {
    const settingsVisWidget = findWidget(node, "settings_visibility");
    const inputModeWidget = findWidget(node, "input_mode");
    const countWidget = findWidget(node, "lora_count");
    if (!settingsVisWidget || !inputModeWidget || !countWidget) return;

    const isSimple = settingsVisWidget.value === "simple";
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
        toggleWidget(node, findWidget(node, `conflict_mode_${i}`), visible && !isSimple);
        toggleWidget(node, findWidget(node, `key_filter_${i}`), visible && !isSimple);
    }

    // Hide base_model_filter in text mode (only useful for dropdown combos)
    const filterWidget = findWidget(node, "base_model_filter");
    if (filterWidget) {
        toggleWidget(node, filterWidget, !isText);
    }

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
    app.canvas?.setDirty?.(true, true);
}

// --- Base Model Filter (requires ComfyUI-Lora-Manager) ---

const LM_LORAS_LIST_URL = "/api/lm/loras/list";
const LM_BASE_MODELS_URL = "/api/lm/loras/base-models?limit=100";
const PAGE_SIZE = 100;

async function fetchJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}

/**
 * Fetch ALL LoRAs from the Lora Manager (paginating through the 100-per-page cap)
 * and build a Map of relative_path -> base_model for local filtering.
 */
async function buildLoraBaseModelMap(fullLoraList) {
    const map = new Map();
    let page = 1;
    let totalPages = 1;

    do {
        const params = new URLSearchParams({
            page: String(page),
            page_size: String(PAGE_SIZE),
            sort_by: "name",
        });
        const data = await fetchJson(`${LM_LORAS_LIST_URL}?${params}`);
        totalPages = data.total_pages || 1;

        for (const item of data.items || []) {
            if (!item.file_path || !item.base_model) continue;
            const apiPath = item.file_path;
            // Match API absolute path against ComfyUI's relative paths
            for (const relPath of fullLoraList) {
                if (relPath === "None") continue;
                if (
                    apiPath === relPath ||
                    apiPath.endsWith("/" + relPath) ||
                    apiPath.endsWith("\\" + relPath)
                ) {
                    map.set(relPath, item.base_model);
                    break;
                }
            }
        }
        page++;
    } while (page <= totalPages);

    return map;
}

function setComboOptions(widget, options) {
    widget.options.values = options;
    // Do NOT reset the selected value — preserve it even if not in the filtered list.
    // ComfyUI COMBO widgets allow values not in the options list; they just can't be
    // re-selected from the dropdown. This prevents silent data loss when switching filters.
}

async function initBaseModelFilter(node, retries = 0) {
    const filterWidget = findWidget(node, "base_model_filter");
    if (!filterWidget) return;

    // Cache the full LoRA list from the first lora_name widget
    const firstLoraWidget = findWidget(node, "lora_name_1");
    if (!firstLoraWidget) return;
    const loraValues = firstLoraWidget.options?.values;
    if (!loraValues || loraValues.length <= 1) {
        // Widget not yet populated, retry (max 20 attempts = 10 seconds)
        if (retries < 20) {
            setTimeout(() => initBaseModelFilter(node, retries + 1), 500);
        }
        return;
    }
    const fullLoraList = [...loraValues];

    // Try to detect Lora Manager and fetch base models
    let baseModels;
    try {
        const data = await fetchJson(LM_BASE_MODELS_URL);
        baseModels = (data.base_models || [])
            .map((m) => m.name)
            .filter(Boolean);
    } catch {
        // Lora Manager not installed — hide filter widget
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    if (baseModels.length === 0) {
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    // Build the path → base_model map once (handles pagination)
    let loraBaseModelMap;
    try {
        loraBaseModelMap = await buildLoraBaseModelMap(fullLoraList);
    } catch {
        toggleWidget(node, filterWidget, false);
        updateVisibility(node);
        return;
    }

    // Populate filter dropdown
    const filterOptions = ["All", ...baseModels];
    setComboOptions(filterWidget, filterOptions);

    // Apply current filter value (handles workflow restore)
    if (filterWidget.value && filterWidget.value !== "All") {
        applyLoraFilter(node, filterWidget.value, fullLoraList, loraBaseModelMap);
    }

    // Intercept future filter changes
    interceptWidgetValue(filterWidget, (newVal) => {
        applyLoraFilter(node, newVal, fullLoraList, loraBaseModelMap);
    });
}

function applyLoraFilter(node, baseModel, fullLoraList, loraBaseModelMap) {
    const MAX = 10;
    let filteredList;

    if (baseModel === "All") {
        filteredList = fullLoraList;
    } else {
        filteredList = fullLoraList.filter(
            (name) =>
                name === "None" || loraBaseModelMap.get(name) === baseModel
        );
        // If nothing matched, fall back to full list
        if (filteredList.length <= 1) {
            filteredList = fullLoraList;
        }
    }

    for (let i = 1; i <= MAX; i++) {
        const w = findWidget(node, `lora_name_${i}`);
        if (w) setComboOptions(w, filteredList);
    }

    app.canvas?.setDirty?.(true, true);
}

// --- LoRA name display: show filename only, strip directory path ---

function patchLoraDisplayValue(widget) {
    if (!widget || widget._loraDisplayPatched) return;
    widget._loraDisplayPatched = true;

    // Override _displayValue getter to show filename without path
    const proto = Object.getPrototypeOf(widget);
    const desc = Object.getOwnPropertyDescriptor(proto, "_displayValue")
        || Object.getOwnPropertyDescriptor(widget, "_displayValue");

    Object.defineProperty(widget, "_displayValue", {
        configurable: true,
        enumerable: true,
        get() {
            const fullVal = desc?.get ? desc.get.call(widget) : widget.value;
            if (!fullVal || typeof fullVal !== "string") return fullVal;
            // Strip directory path, show filename only
            const filename = fullVal.split("/").pop() || fullVal;
            return filename.split("\\").pop() || filename;
        },
    });
}

function patchLoraWidgets(node) {
    for (const w of node.widgets || []) {
        if (w.type === "combo" && /^lora_name/.test(w.name)) {
            patchLoraDisplayValue(w);
        }
    }
}

// --- Node Registration ---

app.registerExtension({
    name: "LoRAOptimizer.LoRAStackDynamic",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAStackDynamic") return;

        // Intercept settings_visibility, input_mode, and lora_count changes to update visibility
        for (const w of node.widgets || []) {
            if (w.name === "input_mode") {
                interceptWidgetValue(w, (newVal) => {
                    syncLoraValues(node, newVal === "text");
                    updateVisibility(node);
                });
            } else if (w.name === "settings_visibility" || w.name === "lora_count") {
                interceptWidgetValue(w, () => updateVisibility(node));
            }
        }

        // Initial visibility update — delay to ensure widgets are fully initialized
        setTimeout(() => {
            updateVisibility(node);
            // Initialize base model filter after visibility is set
            initBaseModelFilter(node);
            // Patch LoRA combo display to show filename only
            patchLoraWidgets(node);
        }, 100);
    },
});

app.registerExtension({
    name: "LoRAOptimizer.LoRAStack",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAStack") return;
        setTimeout(() => patchLoraWidgets(node), 100);
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
