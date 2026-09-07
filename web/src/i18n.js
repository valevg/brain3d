// Локализация интерфейса (ru/en). Простой словарь + подстановка {параметров} — без
// внешних зависимостей, этого достаточно для объёма текста в этом приложении.

const MESSAGES = {
  ru: {
    "app.title": "Brain3D AI",
    "app.subtitle": "3D-визуализация КТ/МРТ головного мозга",

    "uploader.title": "Исследование",
    "uploader.prompt": "Выберите файл исследования (ZIP с DICOM или NIfTI)",
    "uploader.chooseFile": "Выбрать файл",
    "uploader.noFileChosen": "Файл не выбран",
    "uploader.uploadButton": "Загрузить",
    "uploader.uploading": "Загрузка исследования…",
    "uploader.uploaded": "Загружено: {modality}, срезов: {slices}",
    "uploader.detectingResources": "Определение доступных ресурсов…",
    "uploader.error": "Ошибка загрузки: {message}",
    "uploader.startButton": "Запустить обработку",
    "uploader.starting": "Запуск пайплайна…",

    "resources.title": "Ресурсы системы",
    "resources.detecting": "Определение ресурсов…",
    "resources.gpuAvailable": "GPU: {model} ({memory} ГБ)",
    "resources.gpuUnavailable": "GPU: недоступен (режим CPU)",
    "resources.ram": "Оперативная память: {value} ГБ",
    "resources.disk": "Свободно на диске: {value} ГБ",
    "resources.internetOnline": "Интернет: подключён",
    "resources.internetOffline": "Интернет: недоступен (офлайн-режим)",
    "resources.recommended": "Рекомендуемые модели: {models}",
    "resources.refresh": "Обновить",

    "settings.title": "Настройки пайплайна",
    "settings.modality": "Модальность",
    "settings.modality.auto": "Авто",
    "settings.modality.ct": "КТ",
    "settings.modality.mri": "МРТ",
    "settings.tumorModel": "Модель опухоли",
    "settings.vesselModel": "Модель сосудов",
    "settings.option.auto": "Авто (выбор по ресурсам)",
    "settings.option.monai": "MONAI BraTS (Swin UNETR)",
    "settings.option.vista3d": "NVIDIA VISTA-3D",
    "settings.option.totalsegmentator": "TotalSegmentator",
    "settings.option.frangi": "Фильтр Франги (CPU)",
    "settings.fastMode": "Быстрый режим (менее точные модели)",
    "settings.offlineMode": "Офлайн-режим (не использовать сеть/API)",
    "settings.exportFormat": "Формат экспорта",

    "models.title": "Модели сегментации",
    "models.refresh": "Обновить список",
    "models.download": "Скачать",
    "models.downloading": "Загрузка…",
    "models.clearCache": "Очистить кэш",
    "models.clearCacheConfirm": "Удалить все загруженные модели ({size} ГБ)? Их придётся скачать заново.",
    "models.cacheInfo": "Кэш моделей: {size} ГБ ({count} шт.) — {dir}",
    "models.status.loaded": "Загружена",
    "models.status.not_downloaded": "Не загружена",
    "models.status.downloading": "Загружается…",
    "models.status.error": "Ошибка загрузки",
    "models.status.api_available": "Доступна через API",
    "models.size": "{size} ГБ",
    "models.downloadError": "Не удалось загрузить {name}: {message}",
    "models.tryAlternative": "Попробовать другую модель",
    "models.fallbackHint.tumor": "Если модель недоступна, обработка опухоли может быть пропущена либо будет использован TotalSegmentator (без выделения опухоли отдельно).",
    "models.fallbackHint.vessel": "Если модель недоступна, автоматически будет использован фильтр Франги — работает на CPU и не требует загрузки.",
    "models.fallbackHint.generic": "Пайплайн попробует автоматически переключиться на резервный метод.",
    "models.lastUsed": "Использовалась: {date}",
    "models.useCount": "Использований: {count}",
    "models.empty": "Список моделей пуст",

    "progress.title": "Обработка",
    "progress.stage.loading": "Загрузка DICOM",
    "progress.stage.models": "Выбор и загрузка моделей",
    "progress.stage.preprocessing": "Препроцессинг",
    "progress.stage.segmentation": "Сегментация",
    "progress.stage.mesh": "Построение 3D-моделей",
    "progress.stage.export": "Экспорт",
    "progress.stage.done": "Завершено",
    "progress.usingModels": "Мозг: {brain} · Опухоль: {tumor} · Сосуды: {vessels}",
    "progress.gpuMode": "GPU",
    "progress.cpuMode": "CPU",
    "progress.failed": "Обработка завершилась с ошибкой: {message}",

    "results.title": "Результаты обработки",
    "results.modelsUsed": "Использованные модели",
    "results.stage.brain": "Мозг",
    "results.stage.tumor": "Опухоль",
    "results.stage.vessels": "Сосуды",
    "results.notFound": "не найдено",
    "results.confidence": "уверенность {value}%",
    "results.polygons": "{count} структур(ы), полигонов: {faces}",
    "results.downloadButton": "Скачать 3D-модель",
    "results.viewButton": "Показать в 3D",

    "common.gigabytes": "ГБ",
    "common.close": "Закрыть",
    "common.retry": "Повторить",
    "common.unknown": "неизвестно",
    "common.na": "н/д",
  },
  en: {
    "app.title": "Brain3D AI",
    "app.subtitle": "3D visualization of brain CT/MRI",

    "uploader.title": "Study",
    "uploader.prompt": "Choose a study file (DICOM ZIP or NIfTI)",
    "uploader.chooseFile": "Choose file",
    "uploader.noFileChosen": "No file chosen",
    "uploader.uploadButton": "Upload",
    "uploader.uploading": "Uploading study…",
    "uploader.uploaded": "Uploaded: {modality}, slices: {slices}",
    "uploader.detectingResources": "Detecting available resources…",
    "uploader.error": "Upload error: {message}",
    "uploader.startButton": "Start processing",
    "uploader.starting": "Starting pipeline…",

    "resources.title": "System resources",
    "resources.detecting": "Detecting resources…",
    "resources.gpuAvailable": "GPU: {model} ({memory} GB)",
    "resources.gpuUnavailable": "GPU: unavailable (CPU mode)",
    "resources.ram": "RAM: {value} GB",
    "resources.disk": "Free disk space: {value} GB",
    "resources.internetOnline": "Internet: connected",
    "resources.internetOffline": "Internet: unavailable (offline mode)",
    "resources.recommended": "Recommended models: {models}",
    "resources.refresh": "Refresh",

    "settings.title": "Pipeline settings",
    "settings.modality": "Modality",
    "settings.modality.auto": "Auto",
    "settings.modality.ct": "CT",
    "settings.modality.mri": "MRI",
    "settings.tumorModel": "Tumor model",
    "settings.vesselModel": "Vessel model",
    "settings.option.auto": "Auto (choose by resources)",
    "settings.option.monai": "MONAI BraTS (Swin UNETR)",
    "settings.option.vista3d": "NVIDIA VISTA-3D",
    "settings.option.totalsegmentator": "TotalSegmentator",
    "settings.option.frangi": "Frangi filter (CPU)",
    "settings.fastMode": "Fast mode (less accurate models)",
    "settings.offlineMode": "Offline mode (no network/API)",
    "settings.exportFormat": "Export format",

    "models.title": "Segmentation models",
    "models.refresh": "Refresh list",
    "models.download": "Download",
    "models.downloading": "Downloading…",
    "models.clearCache": "Clear cache",
    "models.clearCacheConfirm": "Delete all downloaded models ({size} GB)? They will need to be re-downloaded.",
    "models.cacheInfo": "Model cache: {size} GB ({count}) — {dir}",
    "models.status.loaded": "Loaded",
    "models.status.not_downloaded": "Not downloaded",
    "models.status.downloading": "Downloading…",
    "models.status.error": "Download error",
    "models.status.api_available": "Available via API",
    "models.size": "{size} GB",
    "models.downloadError": "Failed to download {name}: {message}",
    "models.tryAlternative": "Try another model",
    "models.fallbackHint.tumor": "If unavailable, tumor segmentation may be skipped, or TotalSegmentator will be used (without a separate tumor structure).",
    "models.fallbackHint.vessel": "If unavailable, the Frangi filter will be used automatically — runs on CPU, no download needed.",
    "models.fallbackHint.generic": "The pipeline will try to fall back to an alternative method automatically.",
    "models.lastUsed": "Last used: {date}",
    "models.useCount": "Uses: {count}",
    "models.empty": "No models available",

    "progress.title": "Processing",
    "progress.stage.loading": "Loading DICOM",
    "progress.stage.models": "Selecting and downloading models",
    "progress.stage.preprocessing": "Preprocessing",
    "progress.stage.segmentation": "Segmentation",
    "progress.stage.mesh": "Building 3D meshes",
    "progress.stage.export": "Export",
    "progress.stage.done": "Done",
    "progress.usingModels": "Brain: {brain} · Tumor: {tumor} · Vessels: {vessels}",
    "progress.gpuMode": "GPU",
    "progress.cpuMode": "CPU",
    "progress.failed": "Processing failed: {message}",

    "results.title": "Processing results",
    "results.modelsUsed": "Models used",
    "results.stage.brain": "Brain",
    "results.stage.tumor": "Tumor",
    "results.stage.vessels": "Vessels",
    "results.notFound": "not found",
    "results.confidence": "confidence {value}%",
    "results.polygons": "{count} structure(s), faces: {faces}",
    "results.downloadButton": "Download 3D model",
    "results.viewButton": "Show in 3D",

    "common.gigabytes": "GB",
    "common.close": "Close",
    "common.retry": "Retry",
    "common.unknown": "unknown",
    "common.na": "n/a",
  },
};

const STORAGE_KEY = "brain3d_locale";

function _detectDefaultLocale() {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && MESSAGES[stored]) return stored;
  return navigator.language?.toLowerCase().startsWith("ru") ? "ru" : "en";
}

let currentLocale = _detectDefaultLocale();
const listeners = new Set();

/** Возвращает текущую локаль ("ru" | "en"). */
export function getLocale() {
  return currentLocale;
}

/** Переключает локаль интерфейса и уведомляет подписчиков (см. onLocaleChange). */
export function setLocale(locale) {
  if (!MESSAGES[locale] || locale === currentLocale) return;
  currentLocale = locale;
  localStorage.setItem(STORAGE_KEY, locale);
  listeners.forEach((listener) => listener(locale));
}

/** Подписывается на смену локали; возвращает функцию отписки. */
export function onLocaleChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Переводит ключ сообщения с подстановкой {параметров}.
 * @param {string} key
 * @param {Record<string, string|number>} [params]
 * @returns {string}
 */
export function t(key, params = {}) {
  const template = MESSAGES[currentLocale]?.[key] ?? MESSAGES.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_match, name) => (params[name] ?? `{${name}}`).toString());
}
