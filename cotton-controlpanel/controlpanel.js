/**
 * Cotton Control Panel — drop-in DOM controls for p5 (or any) sketches.
 *
 * In index.html:
 *   <link rel="stylesheet" href="cotton-controlpanel/controlpanel.css">
 *   <script src="cotton-controlpanel/controlpanel.js"></script>
 *
 * In setup():
 *   const panel = CottonControlPanel.create({
 *     parent: '#control-panel',
 *     title: 'Settings',
 *     controls: [
 *       { type: 'slider', id: 'size', min: 1, max: 100, value: 50, label: 'Size', suffix: ' px' },
 *       { type: 'checkbox', id: 'debug', label: 'Debug', value: false },
 *       { type: 'radio', id: 'mode', name: 'draw-mode', value: 'a',
 *         options: [{ value: 'a', label: 'Mode A' }, { value: 'b', label: 'Mode B' }] },
 *       { type: 'select', id: 'palette', label: 'Palette', value: 'warm',
 *         options: [{ value: 'warm', label: 'Warm' }, { value: 'cool', label: 'Cool' }] },
 *       { type: 'file', id: 'texture', accept: 'image/*', label: 'Texture' },
 *       { type: 'number', id: 'count', min: 0, max: 10, value: 3, label: 'Count', inline: true },
 *       { type: 'button', id: 'go', label: 'Run', medium: true, onClick(panel) { ... } },
 *       { type: 'group', className: 'my-row', controls: [ ... ] },
 *       { type: 'fieldset', id: 'adv', legend: 'Advanced', hidden: true, controls: [
 *           { type: 'slider', id: 'detail', min: 0, max: 10, value: 3, label: 'Detail' }
 *         ] }
 *     ]
 *   });
 *   // read in draw: panel.values.size, panel.get('mode'), …
 *
 * By default a hint is shown; clicking it toggles the mount, as does the H key (e.g. #control-panel).
 * On viewports under 1000px wide at create time, default copy is "Tap to hide controls".
 * Disable with hint: false and/or hotkey: false; override copy with hintText.
 */
(function (global) {
  'use strict';

  function resolveParent(parent) {
    if (!parent) return document.body;
    if (typeof parent === 'string') {
      const el = document.querySelector(parent);
      if (!el) console.warn('CottonControlPanel: parent not found:', parent);
      return el || document.body;
    }
    return parent;
  }

  function initialValue(spec) {
    if (spec.value !== undefined) return spec.value;
    if (spec.default !== undefined) return spec.default;
    if (spec.type === 'checkbox') return false;
    if (spec.type === 'radio') return spec.options && spec.options[0] ? spec.options[0].value : '';
    if (spec.type === 'select') return spec.options && spec.options[0] ? spec.options[0].value : '';
    if (spec.type === 'slider') return spec.min != null ? spec.min : 0;
    return null;
  }

  function CottonControlPanel(api) {
    this._values = api.values;
    this._listeners = api.listeners;
    this.root = api.root;
    this.values = api.values;
  }

  CottonControlPanel.prototype.get = function (id) {
    return this._values[id];
  };

  CottonControlPanel.prototype.set = function (id, value) {
    const setter = this._setters[id];
    if (!setter) {
      console.warn('CottonControlPanel: unknown id:', id);
      return;
    }
    setter(value);
  };

  CottonControlPanel.prototype.on = function (id, fn) {
    if (!this._listeners[id]) this._listeners[id] = [];
    this._listeners[id].push(fn);
    const self = this;
    return function unsubscribe() {
      const arr = self._listeners[id];
      if (!arr) return;
      const i = arr.indexOf(fn);
      if (i !== -1) arr.splice(i, 1);
    };
  };

  CottonControlPanel.prototype.addSwatch = function (controlId, swatchSpec) {
    var info = this._swatchRows && this._swatchRows[controlId];
    if (!info) return;
    info.addSwatch(swatchSpec);
  };

  CottonControlPanel.prototype.destroy = function () {
    if (typeof this._hotkeyTeardown === 'function') {
      this._hotkeyTeardown();
      this._hotkeyTeardown = null;
    }
    if (this.root && this.root.parentNode) this.root.parentNode.removeChild(this.root);
    this._listeners = {};
    this._setters = {};
  };

  CottonControlPanel.prototype.setToolbarVisible = function (visible) {
    const el = this._toolbarToggleEl;
    if (!el) return;
    el.style.display = visible ? '' : 'none';
  };

  CottonControlPanel.prototype.toggleToolbar = function () {
    toggleToolbarMount(this._toolbarToggleEl);
  };

  CottonControlPanel.prototype.isToolbarVisible = function () {
    const el = this._toolbarToggleEl;
    if (!el) return true;
    return el.style.display !== 'none';
  };

  function toggleToolbarMount(toggleEl) {
    if (!toggleEl) return;
    const isHidden = toggleEl.style.display === 'none';
    toggleEl.style.display = isHidden ? '' : 'none';
  }

  function attachToolbarHotkey(panel, toggleEl, hotkey) {
    if (!hotkey && hotkey !== 0) return;
    const ch = String(hotkey).toLowerCase();
    if (ch.length !== 1) return;

    function onKeyDown(e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target;
      if (
        t &&
        (t.tagName === 'TEXTAREA' ||
          t.tagName === 'SELECT' ||
          t.isContentEditable ||
          (t.tagName === 'INPUT' &&
            (t.type === 'text' ||
              t.type === 'number' ||
              t.type === 'search' ||
              t.type === 'email' ||
              t.type === 'password' ||
              t.type === 'range' ||
              t.type === 'file' ||
              t.type === 'checkbox' ||
              t.type === 'radio')))
      ) {
        return;
      }
      if (e.key.toLowerCase() !== ch) return;
      e.preventDefault();
      e.stopPropagation();
      toggleToolbarMount(toggleEl);
    }

    window.addEventListener('keydown', onKeyDown, true);
    const prev = panel._hotkeyTeardown;
    panel._hotkeyTeardown = function () {
      window.removeEventListener('keydown', onKeyDown, true);
      if (typeof prev === 'function') prev();
    };
  }

  function emit(listeners, id, value) {
    const list = listeners[id];
    if (!list) return;
    for (let i = 0; i < list.length; i++) {
      try {
        list[i](value, id);
      } catch (e) {
        console.error('CottonControlPanel onChange listener:', e);
      }
    }
  }

  function bind(panel, id, read, write, onUserInput) {
    panel._values[id] = read();
    panel._setters[id] = function (v) {
      write(v);
      panel._values[id] = read();
    };
    if (onUserInput) onUserInput();
  }

  function updateRangeProgressPct(rangeEl) {
    const min = parseFloat(rangeEl.min);
    const max = parseFloat(rangeEl.max);
    const val = parseFloat(rangeEl.value);
    const lo = Number.isFinite(min) ? min : 0;
    const hi = Number.isFinite(max) ? max : 100;
    const v = Number.isFinite(val) ? val : lo;
    const pct = hi <= lo ? 0 : ((v - lo) / (hi - lo)) * 100;
    rangeEl.style.setProperty('--cp-slider-pct', pct + '%');
  }

  function appendSlider(host, spec, panel, listeners) {
    const id = spec.id;
    if (!id) return console.warn('CottonControlPanel: slider needs id');

    const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-slider');
    const labelRow = el(wrap, 'div', 'cotton-cp-labelrow');
    el(labelRow, 'span', 'cotton-cp-label', spec.label || id);
    const valueSpan = el(labelRow, 'span', 'cotton-cp-value', '');
    const input = /** @type {HTMLInputElement} */ (el(wrap, 'input', 'cotton-cp-range'));
    input.type = 'range';
    input.min = String(spec.min != null ? spec.min : 0);
    input.max = String(spec.max != null ? spec.max : 100);
    if (spec.step != null) input.step = String(spec.step);
    const start = Number(initialValue(spec));
    input.value = String(start);

    function suffix() {
      return spec.suffix || '';
    }
    function syncLabel() {
      valueSpan.textContent = input.value + suffix();
    }

    function read() {
      const n = spec.step != null && String(spec.step).includes('.')
        ? parseFloat(input.value)
        : parseInt(input.value, 10);
      return Number.isFinite(n) ? n : 0;
    }
    function write(v) {
      input.value = String(v);
      syncLabel();
      updateRangeProgressPct(input);
    }

    syncLabel();
    updateRangeProgressPct(input);
    bind(
      panel,
      id,
      read,
      write,
      function () {
        input.addEventListener('input', function () {
          updateRangeProgressPct(input);
          panel._values[id] = read();
          syncLabel();
          if (spec.onChange) spec.onChange(panel._values[id]);
          emit(listeners, id, panel._values[id]);
        });
      }
    );
  }

  function appendCheckbox(host, spec, panel, listeners) {
    const id = spec.id;
    if (!id) return console.warn('CottonControlPanel: checkbox needs id');

    let checkClass = 'cotton-cp-row cotton-cp-check';
    if (spec.toggle) checkClass += ' cotton-cp-check--toggle';
    const label = el(host, 'label', checkClass);
    const input = /** @type {HTMLInputElement} */ (el(label, 'input', spec.toggle ? 'cotton-cp-toggle-input' : null));
    input.type = 'checkbox';
    if (spec.id) input.id = 'cotton-cp-' + spec.id;
    el(label, 'span', 'cotton-cp-check-text', spec.label || id);
    input.checked = !!initialValue(spec);

    function read() {
      return input.checked;
    }
    function write(v) {
      input.checked = !!v;
    }

    bind(panel, id, read, write, function () {
      input.addEventListener('change', function () {
        panel._values[id] = read();
        if (spec.onChange) spec.onChange(panel._values[id]);
        emit(listeners, id, panel._values[id]);
      });
    });
  }

  function appendRadio(host, spec, panel, listeners) {
    const id = spec.id;
    const name = spec.name || id;
    if (!id) return console.warn('CottonControlPanel: radio group needs id');
    if (!spec.options || !spec.options.length) return console.warn('CottonControlPanel: radio needs options');

    const wrap = el(host, 'div', 'cotton-cp-radio-group');
    wrap.setAttribute('role', 'radiogroup');
    if (spec.ariaLabel || spec.groupLabel) wrap.setAttribute('aria-label', spec.ariaLabel || spec.groupLabel);
    const def = String(initialValue(spec));
    const inputs = [];

    spec.options.forEach(function (opt, i) {
      const lab = el(wrap, 'label', 'cotton-cp-row cotton-cp-radio');
      const inp = /** @type {HTMLInputElement} */ (el(lab, 'input', null));
      inp.type = 'radio';
      inp.name = 'cotton-cp-' + name;
      inp.value = String(opt.value);
      if (opt.id) inp.id = opt.id;
      else inp.id = 'cotton-cp-' + id + '-' + i;
      if (String(opt.value) === def) inp.checked = true;
      inputs.push(inp);
      el(lab, 'span', 'cotton-cp-radio-text', opt.label != null ? opt.label : String(opt.value));
    });

    function read() {
      for (let i = 0; i < inputs.length; i++) {
        if (inputs[i].checked) return inputs[i].value;
      }
      return inputs[0] ? inputs[0].value : '';
    }
    function write(v) {
      const s = String(v);
      for (let i = 0; i < inputs.length; i++) {
        inputs[i].checked = inputs[i].value === s;
      }
    }

    if (!inputs.some(function (inp) {
      return inp.checked;
    })) inputs[0].checked = true;

    bind(panel, id, read, write, function () {
      wrap.addEventListener('change', function () {
        panel._values[id] = read();
        if (spec.onChange) spec.onChange(panel._values[id]);
        emit(listeners, id, panel._values[id]);
      });
    });
  }

  function appendSelect(host, spec, panel, listeners) {
    const id = spec.id;
    if (!id) return console.warn('CottonControlPanel: select needs id');

    const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-select');
    if (spec.label) el(wrap, 'span', 'cotton-cp-label', spec.label);
    const select = /** @type {HTMLSelectElement} */ (el(wrap, 'select', 'cotton-cp-select-el'));
    (spec.options || []).forEach(function (opt) {
      const o = /** @type {HTMLOptionElement} */ (document.createElement('option'));
      o.value = String(opt.value);
      o.textContent = opt.label != null ? opt.label : String(opt.value);
      select.appendChild(o);
    });
    select.value = String(initialValue(spec));

    function read() {
      return select.value;
    }
    function write(v) {
      select.value = String(v);
    }

    bind(panel, id, read, write, function () {
      select.addEventListener('change', function () {
        panel._values[id] = read();
        if (spec.onChange) spec.onChange(panel._values[id]);
        emit(listeners, id, panel._values[id]);
      });
    });
  }

  function appendNumber(host, spec, panel, listeners) {
    const id = spec.id;
    if (!id) return console.warn('CottonControlPanel: number needs id');

    const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-number' + (spec.inline ? ' cotton-cp-number-inline' : ''));
    if (spec.label) el(wrap, 'span', 'cotton-cp-label', spec.label);
    const input = /** @type {HTMLInputElement} */ (el(wrap, 'input', 'cotton-cp-number-input'));
    input.type = 'number';
    if (spec.min != null) input.min = String(spec.min);
    if (spec.max != null) input.max = String(spec.max);
    if (spec.step != null) input.step = String(spec.step);
    const start = Number(initialValue(spec));
    input.value = String(Number.isFinite(start) ? start : 0);

    function read() {
      const n = spec.step != null && String(spec.step).includes('.')
        ? parseFloat(input.value)
        : parseInt(input.value, 10);
      return Number.isFinite(n) ? n : 0;
    }
    function write(v) {
      input.value = String(v);
    }

    bind(panel, id, read, write, function () {
      function fire() {
        panel._values[id] = read();
        if (spec.onChange) spec.onChange(panel._values[id]);
        emit(listeners, id, panel._values[id]);
      }
      input.addEventListener('change', fire);
      input.addEventListener('input', fire);
    });
  }

  function appendButton(host, spec, panel, listeners) {
    const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-actions');
    let btnClass = 'cotton-cp-button';
    if (spec.variant === 'primary') btnClass += ' cotton-cp-button--primary';
    else if (spec.variant === 'secondary') btnClass += ' cotton-cp-button--secondary';
    if (spec.block || spec.fullWidth) btnClass += ' cotton-cp-button--block';
    if (spec.bold) btnClass += ' cotton-cp-button--bold';
    else if (spec.medium) btnClass += ' cotton-cp-button--medium';
    const btn = el(wrap, 'button', btnClass, spec.label || 'OK');
    btn.type = 'button';
    if (spec.id) btn.id = spec.id;
    btn.addEventListener('click', function () {
      if (spec.onClick) spec.onClick(panel);
    });
  }

  function appendGroup(host, spec, panel, listeners) {
    const div = el(host, 'div', spec.className || 'cotton-cp-group');
    if (spec.id) div.id = spec.id;
    (spec.controls || []).forEach(function (c) {
      appendControl(div, c, panel, listeners);
    });
  }

  function appendFile(host, spec, panel, listeners) {
    const id = spec.id;
    if (!id) return console.warn('CottonControlPanel: file needs id');

    const input = /** @type {HTMLInputElement} */ (document.createElement('input'));
    input.type = 'file';
    if (spec.accept) input.accept = spec.accept;
    input.id = 'cotton-cp-' + id + '-input';
    input.className = 'cotton-cp-file-input-hidden';

    panel._values[id] = null;
    panel._setters[id] = function () {
      console.warn('CottonControlPanel: file inputs cannot be set programmatically');
    };

    function onFileChange() {
      const files = input.files;
      panel._values[id] = files && files.length ? files[0] : null;
      if (spec.onChange) spec.onChange(panel._values[id], files);
      emit(listeners, id, panel._values[id]);
    }
    input.addEventListener('change', onFileChange);

    if (spec.buttonLabel) {
      const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-file cotton-cp-file--button');
      const btn = el(wrap, 'button', 'cotton-cp-button cotton-cp-button--secondary cotton-cp-button--block', spec.buttonLabel);
      btn.type = 'button';
      btn.addEventListener('click', function () {
        input.click();
      });
      wrap.appendChild(input);
    } else {
      const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-file');
      if (spec.label) el(wrap, 'span', 'cotton-cp-label', spec.label);
      input.className = 'cotton-cp-file-input';
      wrap.appendChild(input);
    }
  }

  function appendColor(host, spec, panel, listeners) {
    const id = spec.id;
    if (!id) return console.warn('CottonControlPanel: color needs id');

    const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-color');
    if (spec.label) el(wrap, 'span', 'cotton-cp-label', spec.label);
    const input = /** @type {HTMLInputElement} */ (el(wrap, 'input', 'cotton-cp-color-input'));
    input.type = 'color';
    input.value = String(initialValue(spec) || '#000000');

    function read() { return input.value; }
    function write(v) { input.value = String(v); }

    bind(panel, id, read, write, function () {
      input.addEventListener('input', function () {
        panel._values[id] = read();
        if (spec.onChange) spec.onChange(panel._values[id]);
        emit(listeners, id, panel._values[id]);
      });
    });
  }

  /**
   * Swatches control: a row of clickable color circles.
   *
   * spec.swatches      — array of { value, color, label?, multicolor?, colors?, thumbnail? }
   * spec.customButton  — adds a "+" swatch that opens a native color picker
   * spec.uploadButton  — adds an "↑" swatch that opens a file picker
   * spec.uploadAccept  — accept attribute for the file picker (default 'image/*')
   * spec.onCustomColor — called with hex string when custom color is picked
   * spec.onUpload      — called with (File, controlId) when a file is chosen
   *
   * panel.addSwatch(controlId, swatchSpec) — dynamically adds a swatch to an existing row
   */
  function appendSwatches(host, spec, panel, listeners) {
    const id = spec.id;
    if (!id) return console.warn('CottonControlPanel: swatches needs id');

    const wrap = el(host, 'div', 'cotton-cp-row cotton-cp-swatches');
    const row = el(wrap, 'div', 'cotton-cp-swatches-row');

    var swatchEls = [];
    var actionBtns = []; // custom + upload buttons — new swatches insert before these
    var def = String(initialValue(spec));
    var customBtn = null;

    function setActive(val) {
      panel._values[id] = val;
      for (var s = 0; s < swatchEls.length; s++) {
        swatchEls[s].el.classList.toggle('cotton-cp-swatch--active', swatchEls[s].value === val);
      }
      if (customBtn) customBtn.classList.toggle('cotton-cp-swatch--active', val === '__custom__');
      if (spec.onChange) spec.onChange(val);
      emit(listeners, id, val);
    }

    function addSwatchToRow(sw) {
      var btn = document.createElement('button');
      btn.className = 'cotton-cp-swatch';
      btn.type = 'button';
      btn.title = sw.label || sw.value;
      btn.setAttribute('aria-label', sw.label || sw.value);
      if (sw.multicolor && sw.colors) {
        btn.classList.add('cotton-cp-swatch--multi');
        btn.style.background = 'conic-gradient(' + sw.colors.join(', ') + ', ' + sw.colors[0] + ')';
      } else if (sw.thumbnail) {
        btn.classList.add('cotton-cp-swatch--thumb');
        btn.style.backgroundImage = 'url("' + sw.thumbnail.replace(/"/g, '\\"') + '")';
        btn.style.backgroundSize = 'cover';
        btn.style.backgroundPosition = 'center';
      } else {
        btn.style.background = sw.color || sw.value;
        if (sw.color === '#FFFFFF' || sw.color === '#ffffff') {
          btn.classList.add('cotton-cp-swatch--light');
        }
      }
      if (String(sw.value) === panel._values[id]) btn.classList.add('cotton-cp-swatch--active');
      btn.addEventListener('click', function () { setActive(sw.value); });
      swatchEls.push({ el: btn, value: sw.value });

      // Insert before action buttons so they stay at the end
      var firstAction = actionBtns[0] || null;
      if (firstAction) row.insertBefore(btn, firstAction);
      else row.appendChild(btn);
      return btn;
    }

    (spec.swatches || []).forEach(function (sw) { addSwatchToRow(sw); });

    // Custom color button
    if (spec.customButton) {
      customBtn = el(row, 'button', 'cotton-cp-swatch cotton-cp-swatch--custom');
      customBtn.type = 'button';
      customBtn.title = 'Custom color';
      customBtn.setAttribute('aria-label', 'Custom color');
      customBtn.textContent = '+';
      actionBtns.push(customBtn);
      var customInput = /** @type {HTMLInputElement} */ (document.createElement('input'));
      customInput.type = 'color';
      customInput.className = 'cotton-cp-file-input-hidden';
      customInput.value = spec.customDefault || '#FF0000';
      wrap.appendChild(customInput);
      customBtn.addEventListener('click', function () { customInput.click(); });
      customInput.addEventListener('input', function () {
        customBtn.style.background = customInput.value;
        customBtn.textContent = '';
        setActive('__custom__');
        if (spec.onCustomColor) spec.onCustomColor(customInput.value);
      });
      if (def === '__custom__') {
        customBtn.classList.add('cotton-cp-swatch--active');
        customBtn.style.background = spec.customDefault || '#FF0000';
        customBtn.textContent = '';
      }
    }

    // Upload button
    if (spec.uploadButton) {
      var uploadBtn = el(row, 'button', 'cotton-cp-swatch cotton-cp-swatch--upload');
      uploadBtn.type = 'button';
      uploadBtn.title = 'Upload texture';
      uploadBtn.setAttribute('aria-label', 'Upload texture');
      uploadBtn.textContent = '\u2191'; // ↑
      actionBtns.push(uploadBtn);
      var fileInput = /** @type {HTMLInputElement} */ (document.createElement('input'));
      fileInput.type = 'file';
      fileInput.accept = spec.uploadAccept || 'image/*';
      fileInput.className = 'cotton-cp-file-input-hidden';
      wrap.appendChild(fileInput);
      uploadBtn.addEventListener('click', function () { fileInput.click(); });
      fileInput.addEventListener('change', function () {
        if (fileInput.files && fileInput.files[0]) {
          if (spec.onUpload) spec.onUpload(fileInput.files[0], id);
          fileInput.value = '';
        }
      });
    }

    panel._values[id] = def;
    panel._setters[id] = function (v) { setActive(v); };

    // Store row reference so addSwatch works later
    if (!panel._swatchRows) panel._swatchRows = {};
    panel._swatchRows[id] = { addSwatch: addSwatchToRow, setActive: setActive };
  }

  function appendSection(host, spec) {
    if (spec.label != null) {
      var div = el(host, 'div', 'cotton-cp-section', spec.label);
      if (spec.id) div.id = spec.id;
    }
  }

  function appendControl(host, spec, panel, listeners) {
    if (!spec || !spec.type) return;
    switch (spec.type) {
      case 'slider':
        appendSlider(host, spec, panel, listeners);
        break;
      case 'checkbox':
        appendCheckbox(host, spec, panel, listeners);
        break;
      case 'radio':
        appendRadio(host, spec, panel, listeners);
        break;
      case 'select':
        appendSelect(host, spec, panel, listeners);
        break;
      case 'file':
        appendFile(host, spec, panel, listeners);
        break;
      case 'number':
        appendNumber(host, spec, panel, listeners);
        break;
      case 'button':
        appendButton(host, spec, panel, listeners);
        break;
      case 'group':
        appendGroup(host, spec, panel, listeners);
        break;
      case 'fieldset':
        appendFieldset(host, spec, panel, listeners);
        break;
      case 'color':
        appendColor(host, spec, panel, listeners);
        break;
      case 'swatches':
        appendSwatches(host, spec, panel, listeners);
        break;
      case 'section':
        appendSection(host, spec);
        break;
      default:
        console.warn('CottonControlPanel: unknown type:', spec.type);
    }
  }

  function appendFieldset(host, spec, panel, listeners) {
    let cls = 'cotton-cp-fieldset';
    if (spec.plain) cls += ' cotton-cp-fieldset--plain';
    if (spec.className) cls += ' ' + spec.className;
    const fs = el(host, 'fieldset', cls);
    if (spec.id) fs.id = spec.id;
    if (spec.hidden) fs.style.display = 'none';
    if (spec.legend) el(fs, 'legend', 'cotton-cp-legend', spec.legend);
    (spec.controls || []).forEach(function (c) {
      appendControl(fs, c, panel, listeners);
    });
  }

  function el(parent, tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null && text !== '') node.textContent = text;
    parent.appendChild(node);
    return node;
  }

  function create(options) {
    options = options || {};
    const parent = resolveParent(options.parent);
    const values = {};
    const listeners = {};
    const panel = new CottonControlPanel({ values: values, listeners: listeners, root: null });
    panel._setters = {};

    const rootClass = (options.className ? options.className + ' ' : '') + 'cotton-cp';
    const root = el(parent, 'div', rootClass.trim());
    if (options.id) root.id = options.id;
    panel.root = root;

    let hintEl = null;
    if (options.hint !== false) {
      const hintText =
        options.hintText != null
          ? options.hintText
          : typeof window !== 'undefined' && window.innerWidth < 1000
            ? 'Tap to hide controls'
            : 'Press \u2018h\u2019 to show/hide controls';
      hintEl = el(root, 'div', 'cotton-cp-hint', hintText);
    }

    const stackBody = el(root, 'div', 'cotton-cp-stack-body cotton-cp--stack');

    if (options.title) el(stackBody, 'h3', 'cotton-cp-title', options.title);

    (options.controls || []).forEach(function (c) {
      appendControl(stackBody, c, panel, listeners);
    });

    const hotkey = options.hotkey !== undefined ? options.hotkey : 'h';
    if (hotkey) {
      const useRoot =
        options.hotkeyToggle === 'root' || parent === document.body;
      panel._toolbarToggleEl = useRoot ? root : parent;
      attachToolbarHotkey(panel, panel._toolbarToggleEl, hotkey);
    }

    if (hintEl && panel._toolbarToggleEl) {
      function onHintClick(e) {
        e.preventDefault();
        toggleToolbarMount(panel._toolbarToggleEl);
      }
      hintEl.addEventListener('click', onHintClick);
      const prevTeardown = panel._hotkeyTeardown;
      panel._hotkeyTeardown = function () {
        hintEl.removeEventListener('click', onHintClick);
        if (typeof prevTeardown === 'function') prevTeardown();
      };
    }

    return panel;
  }

  global.CottonControlPanel = { create: create };
})(typeof window !== 'undefined' ? window : globalThis);
