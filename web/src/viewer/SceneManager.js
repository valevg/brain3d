// Управление сценой Three.js: камера, свет, загрузка и отображение GLB-сцены структур мозга.
//
// core.mesh_generator.MeshGenerator.export_scene() для формата GLB сохраняет ВСЕ
// структуры в ОДИН файл (по одному именованному узлу на структуру: "brain", "tumor",
// "vessel", "hemorrhage" и т.п.) — поэтому здесь загружается один комбинированный файл,
// а не отдельный GLB на структуру, как в прежней версии клиента.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export class SceneManager {
  /**
   * @param {HTMLElement} container - DOM-элемент, в который монтируется canvas.
   */
  constructor(container) {
    this.container = container;
    this.loader = new GLTFLoader();
    this.structures = new Map(); // имя структуры -> THREE.Object3D
    this.sceneRoot = null; // корневой узел последней загруженной сцены

    this._initScene();
    this._initRenderLoop();
    window.addEventListener("resize", () => this._onResize());
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    const { clientWidth, clientHeight } = this.container;
    this.camera = new THREE.PerspectiveCamera(45, clientWidth / clientHeight, 0.1, 5000);
    this.camera.position.set(0, 0, 300);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(clientWidth, clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(1, 1, 1);
    this.scene.add(directional);
  }

  _initRenderLoop() {
    const animate = () => {
      requestAnimationFrame(animate);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }

  _onResize() {
    const { clientWidth, clientHeight } = this.container;
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight);
  }

  /**
   * Загружает комбинированную GLB-сцену (все структуры сразу) и заменяет ею текущую.
   * @param {string} glbUrl - URL, см. api/client.js resolveResultDownloadUrl().
   * @returns {Promise<string[]>} имена найденных в сцене структур.
   */
  async loadCombinedScene(glbUrl) {
    return new Promise((resolve, reject) => {
      this.loader.load(
        glbUrl,
        (gltf) => {
          this.clearScene();

          const root = gltf.scene;
          const foundStructures = [];

          root.traverse((child) => {
            if (!child.isMesh) return;
            child.material.vertexColors = true;
            child.material.transparent = true;

            // Имя структуры — на самом меше либо на его родительском узле
            // (см. core.mesh_generator.MeshGenerator.build_scene: node_name=geom_name=name).
            const structureName = child.name || child.parent?.name;
            if (structureName) {
              this.structures.set(structureName, child);
              foundStructures.push(structureName);
            }
          });

          this.scene.add(root);
          this.sceneRoot = root;
          this._centerCameraOn(root);
          resolve(foundStructures);
        },
        undefined,
        (error) => reject(error)
      );
    });
  }

  /** Удаляет текущую сцену (перед загрузкой новой). */
  clearScene() {
    if (this.sceneRoot) {
      this.scene.remove(this.sceneRoot);
      this.sceneRoot = null;
    }
    this.structures.clear();
  }

  /** Переключает видимость структуры (для чекбоксов "мозг / опухоль / сосуды" в UI). */
  setStructureVisible(structureName, visible) {
    const object = this.structures.get(structureName);
    if (object) object.visible = visible;
  }

  /** Имена структур, реально присутствующих в загруженной сцене. */
  getStructureNames() {
    return Array.from(this.structures.keys());
  }

  _centerCameraOn(object) {
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();

    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(0, 0, size || 300));
    this.camera.updateProjectionMatrix();
  }
}
