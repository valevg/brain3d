// Панель информации о модели: итоги обработки — какие модели использовались,
// время каждого этапа, уверенность сегментации (где доступна) и число полигонов.
//
// Отображает РЕАЛЬНУЮ структуру ответа BrainModelPipeline.process_with_progress()
// (см. PipelineStatusResponse.result в api/schemas/models.py) — уверенность
// сегментации на сегодня считается только для сосудов (VesselSegmenter.segment_adaptive);
// для мозга/опухоли это поле сознательно не показывается, т.к. бэкенд его не считает.

import { resolveResultDownloadUrl } from "../api/client.js";
import { t } from "../i18n.js";

const STAGE_LABELS = [
  ["loading", "progress.stage.loading"],
  ["model_selection", "progress.stage.models"],
  ["preprocessing", "progress.stage.preprocessing"],
  ["brain_segmentation", "results.stage.brain"],
  ["tumor_segmentation", "results.stage.tumor"],
  ["vessel_segmentation", "results.stage.vessels"],
  ["mesh_generation", "progress.stage.mesh"],
  ["export", "progress.stage.export"],
];

export class ModelInfoPanel {
  /** @param {HTMLElement} root */
  constructor(root) {
    this.root = root;
    this.taskId = null;
    this.onViewInScene = null; // (taskId) => Promise<void>, задаётся снаружи (main.js)
    this.root.hidden = true;
  }

  /**
   * Отображает результат завершённой обработки.
   * @param {string} taskId
   * @param {object} result - PipelineStatusResponse.result.
   */
  show(taskId, result) {
    this.taskId = taskId;
    this.root.hidden = false;

    if (result.status !== "ok") {
      this.root.innerHTML = `<h2>${t("results.title")}</h2><p class="error-text">${result.error ?? ""}</p>`;
      return;
    }

    // "Показать в 3D" работает только для GLB — единственного формата, который грузит
    // GLTFLoader на клиенте; для stl/obj/ply результат отдаётся только на скачивание.
    const exportPath = result.export?.path;
    const isGlbViewable = typeof exportPath === "string" && exportPath.toLowerCase().endsWith(".glb");

    this.root.innerHTML = `
      <h2>${t("results.title")}</h2>
      ${this._renderModelsUsed(result.model_info)}
      ${this._renderTiming(result)}
      ${this._renderVesselConfidence(result.vessel_segmentation)}
      ${this._renderMeshSummary(result.mesh_generation)}
      <div class="panel-actions">
        ${isGlbViewable ? `<button type="button" class="btn btn-secondary" data-role="view">${t("results.viewButton")}</button>` : ""}
        <a class="btn btn-primary" data-role="download" href="${resolveResultDownloadUrl(taskId)}" target="_blank" rel="noopener">
          ${t("results.downloadButton")}
        </a>
      </div>
    `;

    this.root.querySelector('[data-role="view"]')?.addEventListener("click", () => {
      this.onViewInScene?.(taskId);
    });
  }

  hide() {
    this.root.hidden = true;
  }

  _renderModelsUsed(modelInfo) {
    if (!modelInfo) return "";
    const rows = Object.entries(modelInfo)
      .map(([task, info]) => {
        const label = t(`results.stage.${task === "vessels" ? "vessels" : task}`) || task;
        const cachedBadge = info.cached ? "💾" : "☁️";
        return `<tr><td>${label}</td><td>${info.strategy}</td><td>${cachedBadge} ${info.source ?? ""}</td></tr>`;
      })
      .join("");

    return `
      <section class="result-section">
        <h3>${t("results.modelsUsed")}</h3>
        <table class="result-table"><tbody>${rows}</tbody></table>
      </section>
    `;
  }

  _renderTiming(result) {
    const rows = STAGE_LABELS.filter(([key]) => result[key]?.elapsed_seconds !== undefined)
      .map(([key, labelKey]) => {
        const seconds = result[key].elapsed_seconds.toFixed(2);
        return `<tr><td>${t(labelKey)}</td><td>${seconds} s</td></tr>`;
      })
      .join("");
    if (!rows) return "";

    return `
      <section class="result-section">
        <h3>⏱ ${t("progress.title")}</h3>
        <table class="result-table"><tbody>${rows}</tbody></table>
      </section>
    `;
  }

  _renderVesselConfidence(vesselResult) {
    if (!vesselResult || typeof vesselResult.confidence !== "number") return "";
    const percent = Math.round(vesselResult.confidence * 100);
    return `
      <section class="result-section">
        <h3>${t("results.stage.vessels")}</h3>
        <p>${vesselResult.method ?? t("common.unknown")} — ${t("results.confidence", { value: percent })}</p>
      </section>
    `;
  }

  _renderMeshSummary(meshResult) {
    if (!meshResult) return "";
    const totalFaces = Object.values(meshResult.face_counts ?? {}).reduce((sum, n) => sum + n, 0);
    return `
      <section class="result-section">
        <h3>${t("progress.stage.mesh")}</h3>
        <p>${t("results.polygons", { count: meshResult.structures.length, faces: totalFaces.toLocaleString() })}</p>
      </section>
    `;
  }
}
