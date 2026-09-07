// Точка входа веб-приложения: связывает панели управления, API-клиент и 3D-сцену.

import "./styles.css";

import { fetchPipelineStatus, resolveResultDownloadUrl, startProcessing, subscribeToProgress } from "./api/client.js";
import { getLocale, onLocaleChange, setLocale, t } from "./i18n.js";
import { ModelInfoPanel } from "./ui/model_info_panel.js";
import { ModelsPanel } from "./ui/models_panel.js";
import { PipelineSettingsPanel } from "./ui/pipeline_settings.js";
import { ProgressBar } from "./ui/progress_bar.js";
import { ResourcesPanel } from "./ui/resources_panel.js";
import { Uploader } from "./ui/uploader.js";
import { SceneManager } from "./viewer/SceneManager.js";

const sceneManager = new SceneManager(document.getElementById("app"));

const resourcesPanel = new ResourcesPanel(document.getElementById("resources-panel"));
const pipelineSettingsPanel = new PipelineSettingsPanel(document.getElementById("pipeline-settings-panel"));
new ModelsPanel(document.getElementById("models-panel")); // самодостаточна, ссылка не нужна
const progressBar = new ProgressBar(document.getElementById("progress-bar-panel"));
const modelInfoPanel = new ModelInfoPanel(document.getElementById("model-info-panel"));

modelInfoPanel.onViewInScene = async (taskId) => {
  try {
    await sceneManager.loadCombinedScene(resolveResultDownloadUrl(taskId));
  } catch (error) {
    window.alert(error.message);
  }
};

/**
 * Запускает обработку исследования и ведёт пользователя через прогресс до результата.
 * @param {string} studyId
 * @param {object} settings - см. PipelineSettingsPanel.getSettings().
 */
async function handleStart(studyId, settings) {
  modelInfoPanel.hide();
  progressBar.setComputeMode(Boolean(resourcesPanel.lastResources?.gpu_available));
  progressBar.setSelectedModels(null);
  progressBar.update(0, t("uploader.starting"));

  const { task_id: taskId } = await startProcessing(studyId, settings);

  await new Promise((resolve) => {
    const socket = subscribeToProgress(taskId, {
      onMessage: (data) => {
        if (data.status === "failed") {
          progressBar.showError(data.message);
          resolve();
          return;
        }
        progressBar.update(data.percent, data.message);
        if (data.status === "done") resolve();
      },
      onClose: () => resolve(),
      onError: () => resolve(),
    });
    // На случай, если соединение оборвётся до финального сообщения — статус всё равно
    // будет опрошен через REST ниже (fetchPipelineStatus), WebSocket здесь не единственный
    // источник истины.
    void socket;
  });

  const status = await fetchPipelineStatus(taskId);
  progressBar.setSelectedModels(status.selected_models);

  if (status.result) {
    modelInfoPanel.show(taskId, status.result);

    const exportPath = status.result.export?.path;
    if (status.result.status === "ok" && typeof exportPath === "string" && exportPath.toLowerCase().endsWith(".glb")) {
      await sceneManager.loadCombinedScene(resolveResultDownloadUrl(taskId)).catch((error) => {
        console.error("Не удалось отобразить 3D-сцену:", error);
      });
    }
  }
}

new Uploader(document.getElementById("uploader-panel"), {
  resourcesPanel,
  pipelineSettingsPanel,
  onStart: handleStart,
});

// Первичное определение ресурсов ещё до загрузки исследования — чтобы панель настроек
// сразу показывала разумные значения по умолчанию (см. п.4 ТЗ).
resourcesPanel
  .refresh()
  .then((resources) => pipelineSettingsPanel.applyRecommendations(resources))
  .catch(() => {});

// --- Локализация статических элементов и переключатель языка ---

function applyStaticTranslations() {
  document.getElementById("app-title").textContent = t("app.title");
  document.getElementById("app-subtitle").textContent = t("app.subtitle");
}

function renderLocaleSwitch() {
  const switchEl = document.getElementById("locale-switch");
  switchEl.innerHTML = ["ru", "en"]
    .map(
      (locale) =>
        `<button type="button" data-locale="${locale}" class="${locale === getLocale() ? "active" : ""}">${locale.toUpperCase()}</button>`
    )
    .join("");
  switchEl.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => setLocale(button.dataset.locale));
  });
}

applyStaticTranslations();
renderLocaleSwitch();

// Панели рендерят переведённый текст один раз при создании; полная перерисовка всего
// приложения при смене языка — самый простой надёжный способ не хранить состояние
// перевода отдельно от разметки каждой панели. Загруженная 3D-сцена/исследование
// при этом сбрасываются — это осознанный компромисс ради простоты.
onLocaleChange(() => window.location.reload());
