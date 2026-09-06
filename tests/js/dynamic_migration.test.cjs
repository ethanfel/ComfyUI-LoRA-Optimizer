const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../js/lora_stack_dynamic.js'), 'utf8');
const context = { console };
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('const HIDDEN_TAG'), source.indexOf('function toggleWidget')), context);
const migrate = values => Array.from(context.migrateWidgetsValues(values));
for (const count of [7, 8, 9]) {
    test(`legacy ${count}-widget slots retain names, filter and default H3 layout`, () => {
        const old = ['advanced', 'dropdown', 10];
        for (let i = 0; i < 10; i++) {
            if (count >= 8) old.push(true);
            old.push(`lora-${i}`, '', 1, 1, 1, 'all', 'all');
            if (count >= 9) old.push(false);
        }
        old.push('H3');
        const actual = migrate(old);
        assert.equal(actual.length, 104);
        for (let i = 0; i < 10; i++) assert.equal(actual[4 + i * 9], `lora-${i}`);
        assert.equal(actual[93], 'H3');
        assert.deepEqual(actual.slice(94), Array(10).fill('auto'));
        actual[96] = 'diffsynth';
        assert.deepEqual(migrate(actual), actual);
    });
}
