// Панель "Отображение ресурсов": GPU/RAM/диск/интернет + рекомендованные модели.

import { detectResources } from "../api/client.js";
import { t } from "../i18n.js";

export class ResourcesPanel {
  /** @param {HTMLElement} root */
  constructor(root) {
    this.root = root;
    this.lastResources = null;
    this._render();
  }

  _render() {
    this.root.innerHTML = `
      <h2>${t("resources.title")}</h2>
      <div class="panel-body" data-role="body">
        <p class="muted">${t("resources.detecting")}</p>
      </div>
      <button type="button" class="btn btn-secondary" data-role="refresh">${t("resources.refresh")}</button>
    `;
    this.bodyEl = this.root.querySelector('[data-role="body"]');
    this.root.querySelector('[data-role="refresh"]').addEventListener("click", () => this.refresh());
  }

  /** Перезапрашивает ресурсы с сервера, обновляет отображение и возвращает данные. */
  async refresh() {
    this.bodyEl.innerHTML = `<p class="muted loading-dots">${t("resources.detecting")}</p>`;
    try {
      const resources = await detectResources();
      this.lastResources = resources;
      this._renderResources(resources);
      return resources;
    } catch (error) {
      this.bodyEl.innerHTML = `<p class="error-text">${error.message}</p>`;
      throw error;
    }
  }

  _renderResources(resources) {
    const gpuLine = resources.gpu_available
      ? t("resources.gpuAvailable", {
          model: resources.gpu_model || t("common.unknown"),
          memory: resources.gpu_memory_gb.toFixed(1),
        })
      : t("resources.gpuUnavailable");
    const internetLine = resources.internet_available
      ? t("resources.internetOnline")
      : t("resources.internetOffline");
    const recommended = resources.recommended_models?.length
      ? resources.recommended_models.join(", ")
      : t("common.na");

    this.bodyEl.innerHTML = `
      <ul class="resource-list">
        <li class="${resources.gpu_available ? "status-ok" : "status-warn"}">
          <span class="resource-icon" aria-hidden="true">${resources.gpu_available ? "🖥️" : "💻"}</span>
          ${gpuLine}
        </li>
        <li>${t("resources.ram", { value: resources.ram_gb.toFixed(1) })}</li>
        <li>${t("resources.disk", { value: resources.disk_free_gb.toFixed(1) })}</li>
        <li class="${resources.internet_available ? "status-ok" : "status-warn"}">${internetLine}</li>
      </ul>
      <p class="recommended-models">${t("resources.recommended", { models: recommended })}</p>
    `;
  }
}
