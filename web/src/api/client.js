// Клиент для взаимодействия с Brain3D AI FastAPI-backend (api/routes.py).
//
// Переписан под пайплайн на готовых моделях: вместо одного /studies/segment теперь
// отдельные эндпоинты выбора/загрузки моделей, определения ресурсов и запуска
// пайплайна с отслеживанием прогресса по WebSocket.

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail.detail ? String(detail.detail) : `HTTP ${response.status}`;
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function jsonBody(payload) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

// --- Исследования --------------------------------------------------------- //

/**
 * Загружает файл исследования (zip с DICOM или NIfTI) на сервер.
 * @param {File} file
 * @returns {Promise<{study_id: string, modality: string, num_slices: number|null}>}
 */
export function uploadStudy(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/studies/upload", { method: "POST", body: formData });
}

// --- Ресурсы ---------------------------------------------------------------- //

/**
 * Определяет доступные ресурсы (GPU/RAM/диск/интернет) и рекомендует модели.
 * @returns {Promise<{gpu_available:boolean, gpu_memory_gb:number, gpu_model:string,
 *   ram_gb:number, disk_free_gb:number, internet_available:boolean, recommended_models:string[]}>}
 */
export function detectResources() {
  return request("/api/detect-resources", { method: "POST" });
}

// --- Модели ------------------------------------------------------------------ //

/** @returns {Promise<{models: Array<object>}>} */
export function fetchAvailableModels() {
  return request("/api/models/available");
}

/**
 * Запускает фоновую загрузку модели.
 * @param {string} modelName
 * @returns {Promise<{task_id: string, status: string, websocket_url: string}>}
 */
export function downloadModel(modelName) {
  return request("/api/models/download", jsonBody({ model_name: modelName }));
}

/** @returns {Promise<{total_size_gb:number, cache_dir:string, models:Array<object>}>} */
export function fetchCacheInfo() {
  return request("/api/cache/info");
}

/** @returns {Promise<{freed_gb:number, cache_dir:string}>} */
export function clearModelCache() {
  return request("/api/cache/clear", { method: "POST" });
}

// --- Пайплайн ---------------------------------------------------------------- //

/**
 * Запускает обработку исследования на готовых моделях.
 * @param {string} studyId
 * @param {{modality?:string, tumor_model?:string, vessel_model?:string,
 *   use_fast_mode?:boolean, offline_mode?:boolean, export_format?:string}} options
 * @returns {Promise<{task_id: string, status: string, websocket_url: string}>}
 */
export function startProcessing(studyId, options = {}) {
  return request(`/api/process/${studyId}`, jsonBody(options));
}

/**
 * @param {string} taskId
 * @returns {Promise<{task_id:string, status:string, current_stage:string,
 *   progress_percent:number, message:string|null, selected_models:Record<string,string>,
 *   result:object|null}>}
 */
export function fetchPipelineStatus(taskId) {
  return request(`/api/pipeline/status/${taskId}`);
}

/** Возвращает URL для скачивания результата (GLB/OBJ-сцена или ZIP со STL/PLY). */
export function resolveResultDownloadUrl(taskId) {
  return `${API_BASE_URL}/api/process/${taskId}/download`;
}

/**
 * Подписывается на прогресс задачи (скачивание модели или пайплайн) по WebSocket.
 * @param {string} taskId
 * @param {{onMessage?:(data:{percent:number|null,message:string,status?:string})=>void,
 *   onClose?:()=>void, onError?:(err:Event|Error)=>void}} handlers
 * @returns {WebSocket}
 */
export function subscribeToProgress(taskId, { onMessage, onClose, onError } = {}) {
  const socket = new WebSocket(`${WS_BASE_URL}/ws/progress/${taskId}`);

  socket.addEventListener("message", (event) => {
    try {
      onMessage?.(JSON.parse(event.data));
    } catch (error) {
      onError?.(error);
    }
  });
  socket.addEventListener("close", () => onClose?.());
  socket.addEventListener("error", (event) => onError?.(event));

  return socket;
}
