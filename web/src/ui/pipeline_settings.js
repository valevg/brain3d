// Настройки пайплайна: модальность, модель опухоли/сосудов, fast/offline режимы, формат экспорта.
//
// Значения option'ов tumor-model/vessel-model строго соответствуют ключам
// pipelines.brain_pipeline.TUMOR_STRATEGY_MAP / VESSEL_STRATEGY_MAP на бэкенде —
// именно эти строки принимает POST /api/process/{study_id} (ProcessingRequest).

import { t } from "../i18n.js";

const TUMOR_OPTIONS = [
  { value: "auto", labelKey: "settings.option.auto" },
  { value: "monai", labelKey: "settings.option.monai" },
  { value: "vista3d", labelKey: "settings.option.vista3d" },
  { value: "totalsegmentator", labelKey: "settings.option.totalsegmentator" },
];

const VESSEL_OPTIONS = [
  { value: "auto", labelKey: "settings.option.auto" },
  { value: "frangi", labelKey: "settings.option.frangi" },
  { value: "totalseg", labelKey: "settings.option.totalsegmentator" },
  { value: "vista3d", labelKey: "settings.option.vista3d" },
];

const EXPORT_FORMATS = ["glb", "stl", "obj", "ply"];

function renderOptions(options) {
  return options.map((opt) => `<option value="${opt.value}">${t(opt.labelKey)}</option>`).join("");
}

export class PipelineSettingsPanel {
  /** @param {HTMLElement} root */
  constructor(root) {
    this.root = root;
    this._render();
  }

  _render() {
    this.root.innerHTML = `
      <h2>${t("settings.title")}</h2>
      <label class="field">
        <span>${t("settings.modality")}</span>
        <select data-role="modality">
          <option value="auto">${t("settings.modality.auto")}</option>
          <option value="ct">${t("settings.modality.ct")}</option>
          <option value="mri">${t("settings.modality.mri")}</option>
        </select>
      </label>
      <label class="field" title="${t("settings.tumorModel")}">
        <span>${t("settings.tumorModel")}</span>
        <select data-role="tumor-model">${renderOptions(TUMOR_OPTIONS)}</select>
      </label>
      <label class="field" title="${t("settings.vesselModel")}">
        <span>${t("settings.vesselModel")}</span>
        <select data-role="vessel-model">${renderOptions(VESSEL_OPTIONS)}</select>
      </label>
      <label class="field field-checkbox">
        <input type="checkbox" data-role="fast-mode" />
        <span>${t("settings.fastMode")}</span>
      </label>
      <label class="field field-checkbox">
        <input type="checkbox" data-role="offline-mode" />
        <span>${t("settings.offlineMode")}</span>
      </label>
      <label class="field">
        <span>${t("settings.exportFormat")}</span>
        <select data-role="export-format">
          ${EXPORT_FORMATS.map((f) => `<option value="${f}">${f.toUpperCase()}</option>`).join("")}
        </select>
      </label>
    `;

    this.modalityEl = this.root.querySelector('[data-role="modality"]');
    this.tumorModelEl = this.root.querySelector('[data-role="tumor-model"]');
    this.vesselModelEl = this.root.querySelector('[data-role="vessel-model"]');
    this.fastModeEl = this.root.querySelector('[data-role="fast-mode"]');
    this.offlineModeEl = this.root.querySelector('[data-role="offline-mode"]');
    this.exportFormatEl = this.root.querySelector('[data-role="export-format"]');
  }

  /** Применяет рекомендации ResourcesPanel (например, offline при отсутствии интернета). */
  applyRecommendations({ internet_available: internetAvailable, gpu_available: gpuAvailable } = {}) {
    if (internetAvailable === false) this.offlineModeEl.checked = true;
    if (gpuAvailable === false) this.fastModeEl.checked = true;
  }

  /**
   * Текущие настройки в формате тела запроса POST /api/process/{study_id}.
   * @returns {{modality:string, tumor_model:string, vessel_model:string,
   *   use_fast_mode:boolean, offline_mode:boolean, export_format:string}}
   */
  getSettings() {
    return {
      modality: this.modalityEl.value,
      tumor_model: this.tumorModelEl.value,
      vessel_model: this.vesselModelEl.value,
      use_fast_mode: this.fastModeEl.checked,
      offline_mode: this.offlineModeEl.checked,
      export_format: this.exportFormatEl.value,
    };
  }
}
