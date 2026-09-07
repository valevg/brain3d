// Прогресс-бар обработки: процент + текущая операция + индикатор используемой
// модели (мозг/опухоль/сосуды) и режим GPU/CPU.
//
// Точный текст этапа/операции ("Загрузка DICOM...", "Сегментация завершена" и т.п.)
// формируется на бэкенде (pipelines.brain_pipeline.BrainModelPipeline) и приходит
// уже готовой строкой — здесь он отображается как есть, без попытки заново
// локализовать его на клиенте (сообщения о прогрессе на данный момент — только
// на русском, независимо от выбранного языка интерфейса).

import { t } from "../i18n.js";

export class ProgressBar {
  /** @param {HTMLElement} root */
  constructor(root) {
    this.root = root;
    this._render();
    this.hide();
  }

  _render() {
    this.root.innerHTML = `
      <div class="progress-header">
        <span class="progress-mode" data-role="mode"></span>
        <span class="progress-percent" data-role="percent"></span>
      </div>
      <progress class="progress-bar" data-role="bar" max="100" value="0"></progress>
      <div class="progress-message" data-role="message"></div>
      <div class="progress-models" data-role="models"></div>
    `;
    this.modeEl = this.root.querySelector('[data-role="mode"]');
    this.percentEl = this.root.querySelector('[data-role="percent"]');
    this.barEl = this.root.querySelector('[data-role="bar"]');
    this.messageEl = this.root.querySelector('[data-role="message"]');
    this.modelsEl = this.root.querySelector('[data-role="models"]');
  }

  /** Индикатор GPU/CPU (см. п.2 ТЗ — режим вычислений для текущего запуска). */
  setComputeMode(gpuAvailable) {
    this.modeEl.textContent = gpuAvailable ? `🖥️ ${t("progress.gpuMode")}` : `💻 ${t("progress.cpuMode")}`;
  }

  /** Показывает, какие модели выбраны для мозга/опухоли/сосудов (доступно с ~20%). */
  setSelectedModels(selectedModels) {
    if (!selectedModels || !Object.keys(selectedModels).length) {
      this.modelsEl.textContent = "";
      return;
    }
    this.modelsEl.textContent = t("progress.usingModels", {
      brain: selectedModels.brain ?? t("common.na"),
      tumor: selectedModels.tumor ?? t("common.na"),
      vessels: selectedModels.vessels ?? t("common.na"),
    });
  }

  /**
   * Обновляет прогресс.
   * @param {number|null} percent - null означает "завершено" (финальное сообщение).
   * @param {string} message
   */
  update(percent, message) {
    this.show();
    this.root.classList.remove("progress-failed");

    if (typeof percent === "number") {
      this.barEl.value = percent;
      this.barEl.removeAttribute("data-indeterminate");
      this.percentEl.textContent = `${percent}%`;
    } else {
      this.barEl.setAttribute("data-indeterminate", "true");
      this.percentEl.textContent = t("progress.stage.done");
    }
    this.messageEl.textContent = message ?? "";
  }

  /** Отображает сбой обработки. */
  showError(message) {
    this.show();
    this.root.classList.add("progress-failed");
    this.messageEl.textContent = t("progress.failed", { message });
  }

  show() {
    this.root.hidden = false;
  }

  hide() {
    this.root.hidden = true;
    this.root.classList.remove("progress-failed");
  }
}
