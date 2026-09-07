// Панель управления моделями: список со статусами, скачивание с прогресс-баром,
// информация о кэше, очистка кэша, обработка ошибок загрузки с подсказкой о фолбэке.

import {
  clearModelCache,
  downloadModel,
  fetchAvailableModels,
  fetchCacheInfo,
  subscribeToProgress,
} from "../api/client.js";
import { t } from "../i18n.js";

const STATUS_ICON = {
  loaded: "✅",
  not_downloaded: "⬇️",
  downloading: "⏳",
  error: "⚠️",
  api_available: "☁️",
};

// Задача (ModelSpec.task) -> ключ подсказки о резервном варианте при сбое загрузки.
// Реальная логика фолбэка — на бэкенде (pipelines.brain_pipeline.FallbackHandler),
// здесь только честное объяснение пользователю, что произойдёт.
const TASK_FALLBACK_HINT_KEY = {
  tumor: "models.fallbackHint.tumor",
  vessel: "models.fallbackHint.vessel",
};

function formatDate(isoString) {
  if (!isoString) return null;
  try {
    return new Date(isoString).toLocaleString();
  } catch {
    return isoString;
  }
}

export class ModelsPanel {
  /** @param {HTMLElement} root */
  constructor(root) {
    this.root = root;
    this.models = [];
    this._render();
    this.refresh();
  }

  _render() {
    this.root.innerHTML = `
      <h2>${t("models.title")}</h2>
      <div class="panel-body" data-role="list"></div>
      <div class="cache-info" data-role="cache-info"></div>
      <div class="panel-actions">
        <button type="button" class="btn btn-secondary" data-role="refresh">${t("models.refresh")}</button>
        <button type="button" class="btn btn-danger" data-role="clear-cache">${t("models.clearCache")}</button>
      </div>
    `;
    this.listEl = this.root.querySelector('[data-role="list"]');
    this.cacheInfoEl = this.root.querySelector('[data-role="cache-info"]');
    this.root.querySelector('[data-role="refresh"]').addEventListener("click", () => this.refresh());
    this.root.querySelector('[data-role="clear-cache"]').addEventListener("click", () => this._onClearCache());
  }

  /** Перезапрашивает список моделей и информацию о кэше с сервера. */
  async refresh() {
    this.listEl.innerHTML = `<p class="muted loading-dots">${t("resources.detecting")}</p>`;
    try {
      const [{ models }, cacheInfo] = await Promise.all([fetchAvailableModels(), fetchCacheInfo()]);
      this.models = models;
      this._renderModelList();
      this._renderCacheInfo(cacheInfo);
    } catch (error) {
      this.listEl.innerHTML = `<p class="error-text">${error.message}</p>`;
    }
  }

  _renderCacheInfo(cacheInfo) {
    this.cacheInfoEl.textContent = t("models.cacheInfo", {
      size: cacheInfo.total_size_gb.toFixed(2),
      count: cacheInfo.models.length,
      dir: cacheInfo.cache_dir,
    });
  }

  _renderModelList() {
    if (!this.models.length) {
      this.listEl.innerHTML = `<p class="muted">${t("models.empty")}</p>`;
      return;
    }

    this.listEl.innerHTML = this.models
      .map((model) => this._renderModelRow(model))
      .join("");

    for (const model of this.models) {
      const row = this.listEl.querySelector(`[data-model="${model.name}"]`);
      const downloadButton = row?.querySelector('[data-role="download-button"]');
      downloadButton?.addEventListener("click", () => this._onDownload(model));
    }
  }

  _renderModelRow(model) {
    const canDownload = model.status === "not_downloaded" || model.status === "error";
    const isDownloading = model.status === "downloading";
    const usageInfo = [
      model.use_count ? t("models.useCount", { count: model.use_count }) : null,
      model.last_used ? t("models.lastUsed", { date: formatDate(model.last_used) }) : null,
    ]
      .filter(Boolean)
      .join(" · ");

    return `
      <div class="model-row status-${model.status}" data-model="${model.name}" title="${model.description}">
        <div class="model-row-main">
          <span class="model-icon" aria-hidden="true">${STATUS_ICON[model.status] ?? "❔"}</span>
          <div class="model-row-text">
            <div class="model-name">${model.name}</div>
            <div class="model-meta">
              ${t(`models.status.${model.status}`)} · ${t("models.size", { size: model.size_gb.toFixed(2) })} · ${model.source}
            </div>
            ${usageInfo ? `<div class="model-usage muted">${usageInfo}</div>` : ""}
            ${model.status === "error" ? this._renderErrorHint(model) : ""}
          </div>
        </div>
        <div class="model-row-actions">
          ${
            isDownloading
              ? `<progress class="model-progress" data-role="progress" max="100"></progress>`
              : canDownload
                ? `<button type="button" class="btn btn-primary btn-small" data-role="download-button">${t("models.download")}</button>`
                : ""
          }
        </div>
      </div>
    `;
  }

  _renderErrorHint(model) {
    const hintKey = TASK_FALLBACK_HINT_KEY[model.task] ?? "models.fallbackHint.generic";
    return `<div class="model-hint">${t(hintKey)}</div>`;
  }

  async _onDownload(model) {
    const row = this.listEl.querySelector(`[data-model="${model.name}"]`);
    const actionsEl = row?.querySelector(".model-row-actions");
    if (actionsEl) {
      actionsEl.innerHTML = `<progress class="model-progress" data-role="progress" max="100"></progress>`;
    }
    const progressEl = row?.querySelector('[data-role="progress"]');

    try {
      const { task_id: taskId } = await downloadModel(model.name);

      await new Promise((resolve, reject) => {
        subscribeToProgress(taskId, {
          onMessage: (data) => {
            if (progressEl && typeof data.percent === "number") progressEl.value = data.percent;
            if (data.status === "done") resolve();
            if (data.status === "failed") reject(new Error(data.message));
          },
          onClose: () => resolve(),
          onError: () => reject(new Error(t("common.unknown"))),
        });
      });
    } catch (error) {
      console.error(`${t("models.downloadError", { name: model.name, message: error.message })}`);
    } finally {
      await this.refresh();
    }
  }

  async _onClearCache() {
    try {
      const cacheInfo = await fetchCacheInfo();
      const confirmed = window.confirm(
        t("models.clearCacheConfirm", { size: cacheInfo.total_size_gb.toFixed(2) })
      );
      if (!confirmed) return;

      await clearModelCache();
      await this.refresh();
    } catch (error) {
      window.alert(error.message);
    }
  }
}
