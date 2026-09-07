// Загрузка исследования: файл -> /studies/upload -> автоопределение ресурсов ->
// рекомендация моделей -> кнопка запуска обработки с текущими настройками.

import { uploadStudy } from "../api/client.js";
import { t } from "../i18n.js";

export class Uploader {
  /**
   * @param {HTMLElement} root
   * @param {{resourcesPanel: import("./resources_panel.js").ResourcesPanel,
   *   pipelineSettingsPanel: import("./pipeline_settings.js").PipelineSettingsPanel,
   *   onStart: (studyId: string, settings: object) => Promise<void>}} deps
   */
  constructor(root, { resourcesPanel, pipelineSettingsPanel, onStart }) {
    this.root = root;
    this.resourcesPanel = resourcesPanel;
    this.pipelineSettingsPanel = pipelineSettingsPanel;
    this.onStart = onStart;
    this.studyId = null;
    this._render();
  }

  _render() {
    this.root.innerHTML = `
      <h2>${t("uploader.title")}</h2>
      <p class="muted">${t("uploader.prompt")}</p>
      <div class="file-input-row">
        <label class="btn btn-secondary file-label">
          ${t("uploader.chooseFile")}
          <input type="file" data-role="file-input" accept=".zip,.nii,.nii.gz" hidden />
        </label>
        <span class="file-name" data-role="file-name">${t("uploader.noFileChosen")}</span>
      </div>
      <button type="button" class="btn btn-primary" data-role="upload-button" disabled>
        ${t("uploader.uploadButton")}
      </button>
      <div class="upload-status" data-role="status"></div>
      <button type="button" class="btn btn-primary btn-start" data-role="start-button" hidden>
        ${t("uploader.startButton")}
      </button>
    `;

    this.fileInputEl = this.root.querySelector('[data-role="file-input"]');
    this.fileNameEl = this.root.querySelector('[data-role="file-name"]');
    this.uploadButtonEl = this.root.querySelector('[data-role="upload-button"]');
    this.statusEl = this.root.querySelector('[data-role="status"]');
    this.startButtonEl = this.root.querySelector('[data-role="start-button"]');

    this.fileInputEl.addEventListener("change", () => this._onFileSelected());
    this.uploadButtonEl.addEventListener("click", () => this._onUpload());
    this.startButtonEl.addEventListener("click", () => this._onStart());
  }

  _onFileSelected() {
    const file = this.fileInputEl.files[0];
    this.fileNameEl.textContent = file ? file.name : t("uploader.noFileChosen");
    this.uploadButtonEl.disabled = !file;
  }

  async _onUpload() {
    const file = this.fileInputEl.files[0];
    if (!file) return;

    this.uploadButtonEl.disabled = true;
    this.startButtonEl.hidden = true;
    this.statusEl.classList.remove("error-text");
    this.statusEl.textContent = t("uploader.uploading");

    try {
      const result = await uploadStudy(file);
      this.studyId = result.study_id;
      this.statusEl.textContent = t("uploader.uploaded", {
        modality: result.modality,
        slices: result.num_slices ?? t("common.na"),
      });
      this.startButtonEl.hidden = false;

      // Точка 8 ТЗ: после загрузки — автоопределение ресурсов и рекомендация моделей,
      // с возможностью пользователю изменить их перед запуском (PipelineSettingsPanel
      // остаётся редактируемой — applyRecommendations лишь проставляет значения по умолчанию).
      this.statusEl.textContent += ` ${t("uploader.detectingResources")}`;
      const resources = await this.resourcesPanel.refresh().catch(() => null);
      if (resources) this.pipelineSettingsPanel.applyRecommendations(resources);
    } catch (error) {
      this.statusEl.classList.add("error-text");
      this.statusEl.textContent = t("uploader.error", { message: error.message });
    } finally {
      this.uploadButtonEl.disabled = false;
    }
  }

  async _onStart() {
    if (!this.studyId) return;
    this.startButtonEl.disabled = true;
    this.statusEl.classList.remove("error-text");
    this.statusEl.textContent = t("uploader.starting");
    try {
      await this.onStart?.(this.studyId, this.pipelineSettingsPanel.getSettings());
    } catch (error) {
      this.statusEl.classList.add("error-text");
      this.statusEl.textContent = t("uploader.error", { message: error.message });
    } finally {
      this.startButtonEl.disabled = false;
    }
  }
}
