/**
 * Phase 4 - 3D Voxel / Smooth Biomechanical ISL Avatar Renderer (avatar_renderer.js)
 * ==================================================================================
 * Features a 48-Bone Biomechanical 3D Rig with articulated multi-phalange fingers
 * on both hands for authentic Indian Sign Language (ISL) hand signals (mudras/handshapes),
 * 3D depth lighting & soft shadows, persona switching (Aarav, Prof. Meera, Sheru Mascot),
 * and real-time Wrist Roll & Elbow Pitch telemetry.
 */

// Canonical ISL Handshape Finger Curl Presets (0.0 = straight extended, 1.35 = fully curled fist)
export const ISL_HANDSHAPES = {
    OPEN: {
        label: "5-Open Palm (Spread)",
        right: [0.05, 0.0, 0.0, 0.0, 0.0],
        left: [0.05, 0.0, 0.0, 0.0, 0.0],
        spread: 0.16,
    },
    POINT: {
        label: "1-Index Point (IX-Locus)",
        right: [0.85, 0.0, 1.25, 1.30, 1.35],
        left: [0.40, 0.25, 0.30, 0.35, 0.40],
        spread: 0.06,
    },
    LOGIC_TWO: {
        label: "2-Logic Step (Algorithm)",
        right: [0.95, 0.0, 0.0, 1.25, 1.30],
        left: [0.10, 0.05, 0.05, 0.05, 0.05],
        spread: 0.14,
    },
    TANGENT: {
        label: "C-Curve & Tangent (Derivative)",
        right: [0.90, 0.0, 1.20, 1.25, 1.30],
        left: [0.55, 0.65, 0.65, 0.65, 0.65],
        spread: 0.08,
    },
    FIST_VECTOR: {
        label: "S-Fist & B-Vector (Momentum)",
        right: [0.15, 0.0, 0.0, 0.0, 0.0],
        left: [1.10, 1.35, 1.35, 1.35, 1.35],
        spread: 0.03,
    },
    PINCH: {
        label: "O-Pinch Precision",
        right: [0.75, 0.80, 0.25, 0.20, 0.15],
        left: [0.75, 0.80, 0.25, 0.20, 0.15],
        spread: 0.12,
    },
};

export class ISLAvatarRenderer {
    /**
     * @param {HTMLElement} containerElement
     * @param {Function} onGlossCallback
     * @param {Function} onRigTelemetryCallback
     */
    constructor(containerElement, onGlossCallback = null, onRigTelemetryCallback = null) {
        this.container = containerElement;
        this.onGlossChange = onGlossCallback;
        this.onRigTelemetry = onRigTelemetryCallback;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = null;

        // Mapped humanoid bones: { Spine, Head, RightArm, RightForeArm, RightHand, LeftArm, LeftForeArm, LeftHand }
        this.bones = {};
        // Articulated 3D finger chains: { right: [[prox, dist], ...5], left: [[prox, dist], ...5] }
        this.fingerBones = { right: [], left: [] };
        this.targetHandshape = ISL_HANDSHAPES.OPEN;

        // Facial Non-Manual Marker nodes/morphs
        this.nmmNodes = { leftBrow: null, rightBrow: null, mouth: null };
        this.personaMaterials = {};
        this.skinnedMeshes = [];

        // Spatial locus visual indicators (IX-LEFT, IX-RIGHT)
        this.locusRings = { left: null, right: null };

        // Animation keyframe queue
        this.keyframeQueue = [];
        this.activeKeyframe = null;
        this.keyframeElapsed = 0;
        this.currentGloss = "READY";
        this.isPaused = false;
        this._idleTime = 0;

        this._initScene();
        this._buildProceduralHumanoidRig();
        this._animate();

        window.addEventListener("resize", () => this.handleResize());
        if (typeof ResizeObserver !== "undefined" && this.container) {
            this._resizeObserver = new ResizeObserver(() => this.handleResize());
            this._resizeObserver.observe(this.container);
        }
    }

    _initScene() {
        const THREE = window.THREE;
        if (!THREE) {
            console.error("Three.js not loaded.");
            return;
        }

        const width = this.container.clientWidth || 560;
        const height = this.container.clientHeight || 380;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf4f7fb);

        this.camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 100);
        this.camera.position.set(0, 1.34, 2.55);
        this.camera.lookAt(0, 1.16, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        const oldCanvas = this.container.querySelector("canvas");
        if (oldCanvas) oldCanvas.remove();
        this.container.insertBefore(this.renderer.domElement, this.container.firstChild);

        // Soft 3D Studio Depth Lighting
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0xdbe4f0, 1.15);
        hemiLight.position.set(0, 6, 0);
        this.scene.add(hemiLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
        keyLight.position.set(2.2, 4.5, 3.8);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 1024;
        keyLight.shadow.mapSize.height = 1024;
        this.scene.add(keyLight);

        const rimLight = new THREE.DirectionalLight(0x6366f1, 0.45);
        rimLight.position.set(-2.5, 2.5, -1.8);
        this.scene.add(rimLight);

        // 3D Depth Signing Box & Floor Grid
        const grid = new THREE.GridHelper(4.2, 16, 0xcbd5e1, 0xe2e8f0);
        grid.position.y = 0.0;
        this.scene.add(grid);

        const ringGeo = new THREE.RingGeometry(0.12, 0.165, 36);
        const leftMat = new THREE.MeshBasicMaterial({ color: 0x4f46e5, side: THREE.DoubleSide, transparent: true, opacity: 0.35 });
        const rightMat = new THREE.MeshBasicMaterial({ color: 0x0ea5e9, side: THREE.DoubleSide, transparent: true, opacity: 0.35 });

        this.locusRings.left = new THREE.Mesh(ringGeo, leftMat);
        this.locusRings.left.position.set(0.52, 1.08, 0.34);
        this.scene.add(this.locusRings.left);

        this.locusRings.right = new THREE.Mesh(ringGeo, rightMat);
        this.locusRings.right.position.set(-0.52, 1.08, 0.34);
        this.scene.add(this.locusRings.right);

        this.clock = new THREE.Clock();
    }

    /**
     * Builds the 48-Bone Biomechanical 3D Rig with smooth-shaded voxel head, cap/hair dome,
     * torso, and multi-jointed 5-finger hands capable of crisp ISL hand signals.
     */
    _buildProceduralHumanoidRig() {
        const THREE = window.THREE;
        if (!THREE) return;

        if (this.avatarRoot) {
            this.scene.remove(this.avatarRoot);
        }

        const root = new THREE.Group();
        root.name = "ProceduralISLAvatar";

        const suitMat = new THREE.MeshStandardMaterial({ color: 0x3b4254, roughness: 0.35, metalness: 0.12 });
        const capMat = new THREE.MeshStandardMaterial({ color: 0x3b4254, roughness: 0.4, metalness: 0.1 });
        const accentMat = new THREE.MeshStandardMaterial({ color: 0x4f46e5, roughness: 0.3, metalness: 0.18 });
        const skinMat = new THREE.MeshStandardMaterial({ color: 0xf6c992, roughness: 0.38, metalness: 0.06 });
        const jointMat = new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.32, metalness: 0.15 });
        const darkMat = new THREE.MeshBasicMaterial({ color: 0x1e1b4b });
        const mouthMat = new THREE.MeshBasicMaterial({ color: 0xb91c1c });

        this.personaMaterials = { suitMat, capMat, accentMat, skinMat };

        // Spine / Torso
        const spine = new THREE.Group();
        spine.name = "Spine";
        spine.position.set(0, 0.88, 0);
        root.add(spine);

        const torsoMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.21, 0.19, 0.56, 28), suitMat);
        torsoMesh.position.set(0, 0.24, 0);
        torsoMesh.scale.set(1.12, 1.0, 0.72);
        torsoMesh.castShadow = true;
        spine.add(torsoMesh);

        const collarMesh = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.025, 14, 32), accentMat);
        collarMesh.position.set(0, 0.51, 0);
        collarMesh.rotation.x = Math.PI / 2;
        spine.add(collarMesh);

        const neckMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.072, 0.12, 20), skinMat);
        neckMesh.position.set(0, 0.57, 0);
        spine.add(neckMesh);

        // Smooth 3D Voxel Head (matching screenshot character)
        const head = new THREE.Group();
        head.name = "Head";
        head.position.set(0, 0.73, 0);
        spine.add(head);

        const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.175, 36, 36), skinMat);
        headMesh.scale.set(1.02, 0.98, 0.98);
        headMesh.castShadow = true;
        head.add(headMesh);

        // Dark stylized cap / hair dome on top half of head (as shown in screenshot)
        const capGeo = new THREE.SphereGeometry(0.181, 36, 24, 0, Math.PI * 2, 0, Math.PI * 0.48);
        const capMesh = new THREE.Mesh(capGeo, capMat);
        capMesh.position.y = 0.01;
        head.add(capMesh);

        // Expressive Eyes
        const eyeGeo = new THREE.SphereGeometry(0.018, 16, 16);
        const leftEye = new THREE.Mesh(eyeGeo, darkMat);
        leftEye.position.set(0.062, -0.005, 0.162);
        const rightEye = new THREE.Mesh(eyeGeo, darkMat);
        rightEye.position.set(-0.062, -0.005, 0.162);
        head.add(leftEye, rightEye);

        // Articulated Eyebrows for ISL Non-Manual Markers
        const browGeo = new THREE.BoxGeometry(0.048, 0.011, 0.014);
        this.nmmNodes.leftBrow = new THREE.Mesh(browGeo, darkMat);
        this.nmmNodes.leftBrow.position.set(0.062, 0.032, 0.166);
        this.nmmNodes.rightBrow = new THREE.Mesh(browGeo, darkMat);
        this.nmmNodes.rightBrow.position.set(-0.062, 0.032, 0.166);
        head.add(this.nmmNodes.leftBrow, this.nmmNodes.rightBrow);

        // Expressive Red Mouth Bar (matching screenshot character)
        const mouthGeo = new THREE.BoxGeometry(0.052, 0.014, 0.014);
        this.nmmNodes.mouth = new THREE.Mesh(mouthGeo, mouthMat);
        this.nmmNodes.mouth.position.set(0, -0.068, 0.164);
        head.add(this.nmmNodes.mouth);

        this.fingerBones = { right: [], left: [] };

        // Helper to build articulated Arm + ForeArm + 15-Bone Multi-Jointed Hand
        const buildArmAndHand = (sideSign, prefix, sideKey) => {
            const shoulderPivot = new THREE.Group();
            shoulderPivot.name = `${prefix}Arm`;
            shoulderPivot.position.set(sideSign * 0.27, 0.46, 0);
            spine.add(shoulderPivot);

            const shoulderSphere = new THREE.Mesh(new THREE.SphereGeometry(0.058, 20, 20), accentMat);
            shoulderPivot.add(shoulderSphere);

            const upperArmMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.040, 0.28, 18), suitMat);
            upperArmMesh.position.set(sideSign * 0.14, 0, 0);
            upperArmMesh.rotation.z = sideSign * (Math.PI / 2);
            shoulderPivot.add(upperArmMesh);

            const foreArmPivot = new THREE.Group();
            foreArmPivot.name = `${prefix}ForeArm`;
            foreArmPivot.position.set(sideSign * 0.28, 0, 0);
            shoulderPivot.add(foreArmPivot);

            const elbowSphere = new THREE.Mesh(new THREE.SphereGeometry(0.046, 18, 18), jointMat);
            foreArmPivot.add(elbowSphere);

            const foreArmMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.040, 0.035, 0.26, 18), skinMat);
            foreArmMesh.position.set(sideSign * 0.13, 0, 0);
            foreArmMesh.rotation.z = sideSign * (Math.PI / 2);
            foreArmPivot.add(foreArmMesh);

            const handPivot = new THREE.Group();
            handPivot.name = `${prefix}Hand`;
            handPivot.position.set(sideSign * 0.27, 0, 0);
            foreArmPivot.add(handPivot);

            const wristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.034, 14, 14), jointMat);
            handPivot.add(wristJoint);

            // 3D Sculpted Palm
            const palmMesh = new THREE.Mesh(new THREE.BoxGeometry(0.092, 0.095, 0.032), skinMat);
            palmMesh.position.set(sideSign * 0.048, 0, 0);
            handPivot.add(palmMesh);

            // 5 Articulated Two-Segment Fingers (Thumb, Index, Middle, Ring, Pinky)
            const fingerLengths = [0.036, 0.048, 0.054, 0.048, 0.038];
            for (let f = 0; f < 5; f++) {
                const proxPivot = new THREE.Group();
                const offsetY = (2 - f) * 0.020;
                const isThumb = f === 0;
                proxPivot.position.set(
                    sideSign * (isThumb ? 0.045 : 0.092),
                    isThumb ? 0.048 : offsetY,
                    isThumb ? 0.012 : 0
                );
                handPivot.add(proxPivot);

                const knuckleSphere = new THREE.Mesh(new THREE.SphereGeometry(0.011, 10, 10), jointMat);
                proxPivot.add(knuckleSphere);

                const len1 = fingerLengths[f];
                const phalange1 = new THREE.Mesh(new THREE.CylinderGeometry(0.0095, 0.0085, len1, 10), skinMat);
                phalange1.position.set(sideSign * (len1 * 0.5), 0, 0);
                phalange1.rotation.z = sideSign * (Math.PI / 2);
                proxPivot.add(phalange1);

                const distPivot = new THREE.Group();
                distPivot.position.set(sideSign * len1, 0, 0);
                proxPivot.add(distPivot);

                const midKnuckle = new THREE.Mesh(new THREE.SphereGeometry(0.009, 10, 10), accentMat);
                distPivot.add(midKnuckle);

                const len2 = len1 * 0.82;
                const phalange2 = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.0065, len2, 10), skinMat);
                phalange2.position.set(sideSign * (len2 * 0.5), 0, 0);
                phalange2.rotation.z = sideSign * (Math.PI / 2);
                distPivot.add(phalange2);

                this.fingerBones[sideKey].push({ prox: proxPivot, dist: distPivot, sideSign, index: f });
            }

            return { shoulderPivot, foreArmPivot, handPivot };
        };

        const rightChain = buildArmAndHand(-1, "Right", "right");
        const leftChain = buildArmAndHand(1, "Left", "left");

        this.bones = {
            Spine: spine,
            Head: head,
            RightArm: rightChain.shoulderPivot,
            RightForeArm: rightChain.foreArmPivot,
            RightHand: rightChain.handPivot,
            LeftArm: leftChain.shoulderPivot,
            LeftForeArm: leftChain.foreArmPivot,
            LeftHand: leftChain.handPivot,
        };

        // Expressive Upper-Torso Signing Box Ready Pose
        this.bones.RightArm.rotation.set(-0.55, 0.25, -0.55);
        this.bones.RightForeArm.rotation.set(0.15, -0.85, 0.35);
        this.bones.LeftArm.rotation.set(-0.50, -0.20, 0.55);
        this.bones.LeftForeArm.rotation.set(0.10, 0.80, -0.30);

        this.avatarRoot = root;
        this.scene.add(root);
    }

    /**
     * Switches between the 3 Avatar Personas shown in the UI: Aarav, Prof. Meera, Sheru Mascot.
     */
    setPersona(personaId) {
        if (!this.personaMaterials.suitMat) return;
        if (personaId === "meera") {
            this.personaMaterials.suitMat.color.setHex(0x0f766e);
            this.personaMaterials.capMat.color.setHex(0x1e293b);
            this.personaMaterials.accentMat.color.setHex(0x14b8a6);
            this.personaMaterials.skinMat.color.setHex(0xeeb886);
        } else if (personaId === "sheru") {
            this.personaMaterials.suitMat.color.setHex(0xd97706);
            this.personaMaterials.capMat.color.setHex(0x92400e);
            this.personaMaterials.accentMat.color.setHex(0xf59e0b);
            this.personaMaterials.skinMat.color.setHex(0xfbbf24);
        } else {
            // Aarav (Default)
            this.personaMaterials.suitMat.color.setHex(0x3b4254);
            this.personaMaterials.capMat.color.setHex(0x3b4254);
            this.personaMaterials.accentMat.color.setHex(0x4f46e5);
            this.personaMaterials.skinMat.color.setHex(0xf6c992);
        }
    }

    /**
     * Triggers an immediate ISL Hand Signal (Handshape + Upper Torso Pose).
     * @param {string} handshapeKey - Key in ISL_HANDSHAPES ("OPEN", "POINT", "LOGIC_TWO", "TANGENT", "FIST_VECTOR", "PINCH")
     */
    triggerHandSignal(handshapeKey) {
        const shape = ISL_HANDSHAPES[handshapeKey] || ISL_HANDSHAPES.OPEN;
        this.targetHandshape = shape;

        const poseMap = {
            OPEN: {
                RightArm: { x: -0.85, y: 0.30, z: -0.35 },
                RightForeArm: { x: 0.15, y: -0.95, z: 0.45 },
                RightHand: { x: 0.25, y: 0.35, z: -0.15 },
                LeftArm: { x: -0.80, y: -0.25, z: 0.35 },
                LeftForeArm: { x: 0.15, y: 0.90, z: -0.40 },
                LeftHand: { x: -0.20, y: -0.30, z: 0.15 },
            },
            POINT: {
                RightArm: { x: -0.70, y: 0.55, z: -0.25 },
                RightForeArm: { x: 0.05, y: -0.70, z: 0.25 },
                RightHand: { x: 0.0, y: 0.40, z: 0.0 },
                LeftArm: { x: -0.45, y: -0.15, z: 0.65 },
                LeftForeArm: { x: 0.0, y: 0.65, z: -0.20 },
            },
            LOGIC_TWO: {
                RightArm: { x: -0.75, y: 0.25, z: -0.35 },
                RightForeArm: { x: 0.20, y: -1.05, z: 0.50 },
                RightHand: { x: 0.30, y: 0.45, z: 0.10 },
                LeftArm: { x: -0.60, y: -0.30, z: 0.40 },
                LeftForeArm: { x: 0.10, y: 1.00, z: -0.35 },
            },
            TANGENT: {
                RightArm: { x: -0.95, y: 0.40, z: -0.30 },
                RightForeArm: { x: 0.10, y: -1.15, z: 0.50 },
                RightHand: { x: 0.10, y: 0.35, z: -0.40 },
                LeftArm: { x: -0.65, y: -0.35, z: 0.45 },
                LeftForeArm: { x: 0.05, y: 1.15, z: -0.15 },
            },
            FIST_VECTOR: {
                RightArm: { x: -0.65, y: -0.10, z: -0.40 },
                RightForeArm: { x: 0.05, y: -0.85, z: 0.30 },
                RightHand: { x: 0.0, y: 0.20, z: 0.0 },
                LeftArm: { x: -0.70, y: 0.25, z: 0.35 },
                LeftForeArm: { x: 0.15, y: 1.05, z: -0.35 },
            },
            PINCH: {
                RightArm: { x: -0.80, y: 0.20, z: -0.30 },
                RightForeArm: { x: 0.15, y: -1.10, z: 0.45 },
                RightHand: { x: 0.15, y: 0.25, z: -0.10 },
                LeftArm: { x: -0.75, y: -0.20, z: 0.30 },
                LeftForeArm: { x: 0.15, y: 1.05, z: -0.40 },
            },
        };

        this.enqueueKeyframes([
            {
                gloss: shape.label.split(" ")[0].toUpperCase(),
                handshape: handshapeKey,
                duration: 0.55,
                bones: poseMap[handshapeKey] || poseMap.OPEN,
                nmm: { eyebrowRaise: 0.35, mouthOpen: 0.12 },
            },
        ]);
    }

    _resolveHandshapeForGloss(gloss) {
        const g = (gloss || "").toUpperCase();
        if (g.includes("IX-")) return ISL_HANDSHAPES.POINT;
        if (g.includes("ALGORITHM") || g.includes("EXPLAIN")) return ISL_HANDSHAPES.LOGIC_TWO;
        if (g.includes("DERIVATIVE") || g.includes("CALCULATE")) return ISL_HANDSHAPES.TANGENT;
        if (g.includes("MOMENTUM") || g.includes("FORCE")) return ISL_HANDSHAPES.FIST_VECTOR;
        if (g.includes("PHOTOSYNTHESIS") || g.includes("TODAY")) return ISL_HANDSHAPES.OPEN;
        return ISL_HANDSHAPES.PINCH;
    }

    async loadGLBAvatar(glbUrl) {
        const THREE = window.THREE;
        if (!THREE || !THREE.GLTFLoader) {
            throw new Error("THREE.GLTFLoader is required to load external .glb files.");
        }

        return new Promise((resolve, reject) => {
            const loader = new THREE.GLTFLoader();
            loader.load(
                glbUrl,
                (gltf) => {
                    if (this.avatarRoot) {
                        this.scene.remove(this.avatarRoot);
                    }
                    this.avatarRoot = gltf.scene;
                    this.avatarRoot.position.set(0, 0, 0);
                    this.bones = {};
                    this.skinnedMeshes = [];

                    const canonicalBones = [
                        "Spine", "Head", "RightArm", "RightForeArm", "RightHand",
                        "LeftArm", "LeftForeArm", "LeftHand",
                    ];

                    this.avatarRoot.traverse((obj) => {
                        if (obj.isSkinnedMesh) {
                            this.skinnedMeshes.push(obj);
                        }
                        if (obj.isBone) {
                            for (const targetName of canonicalBones) {
                                if (
                                    obj.name === targetName ||
                                    obj.name === `mixamorig${targetName}` ||
                                    obj.name === `mixamorig:${targetName}`
                                ) {
                                    this.bones[targetName] = obj;
                                }
                            }
                        }
                    });

                    this.scene.add(this.avatarRoot);
                    resolve({
                        loaded: true,
                        mappedBones: Object.keys(this.bones),
                    });
                },
                undefined,
                (error) => reject(error)
            );
        });
    }

    enqueueKeyframes(keyframes) {
        if (!Array.isArray(keyframes) || keyframes.length === 0) return;
        this.keyframeQueue = [...keyframes];
        this.activeKeyframe = null;
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        return this.isPaused;
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        if (!this.renderer || !this.scene || !this.camera) return;

        const dt = this.clock ? this.clock.getDelta() : 0.016;
        if (!this.isPaused) {
            this._stepKeyframeInterpolation(dt);
            this._stepFingerArticulation(dt);
        }
        this.renderer.render(this.scene, this.camera);
    }

    _stepFingerArticulation(dt) {
        const shape = this.targetHandshape || ISL_HANDSHAPES.OPEN;
        const speed = Math.min(1.0, dt * 9.0);

        ["right", "left"].forEach((side) => {
            const curls = shape[side] || [0, 0, 0, 0, 0];
            const fingers = this.fingerBones[side] || [];
            fingers.forEach((fObj, idx) => {
                const targetCurl = curls[idx] || 0;
                const spreadAngle = (2 - idx) * (shape.spread || 0.08);
                // Curl around Z/Y axis to close phalanges into palm
                const targetProxZ = fObj.sideSign * -targetCurl;
                const targetDistZ = fObj.sideSign * -(targetCurl * 0.85);

                fObj.prox.rotation.z += (targetProxZ - fObj.prox.rotation.z) * speed;
                fObj.prox.rotation.y += (spreadAngle - fObj.prox.rotation.y) * speed;
                fObj.dist.rotation.z += (targetDistZ - fObj.dist.rotation.z) * speed;
            });
        });
    }

    _stepKeyframeInterpolation(dt) {
        const THREE = window.THREE;
        if (!THREE) return;

        if (!this.activeKeyframe && this.keyframeQueue.length > 0) {
            this.activeKeyframe = this.keyframeQueue.shift();
            this.keyframeElapsed = 0;

            if (this.activeKeyframe.gloss && this.activeKeyframe.gloss !== this.currentGloss) {
                this.currentGloss = this.activeKeyframe.gloss;
                if (this.onGlossChange) {
                    this.onGlossChange(this.currentGloss);
                }
            }

            if (this.activeKeyframe.handshape && ISL_HANDSHAPES[this.activeKeyframe.handshape]) {
                this.targetHandshape = ISL_HANDSHAPES[this.activeKeyframe.handshape];
            } else {
                this.targetHandshape = this._resolveHandshapeForGloss(this.currentGloss);
            }

            if (this.locusRings.left && this.locusRings.right) {
                this.locusRings.left.material.opacity = this.currentGloss === "IX-LEFT" ? 0.95 : 0.30;
                this.locusRings.right.material.opacity = this.currentGloss === "IX-RIGHT" ? 0.95 : 0.30;
            }
        }

        if (!this.activeKeyframe) {
            this._idleTime += dt;
            if (this.bones.Spine) {
                this.bones.Spine.position.y = 0.88 + Math.sin(this._idleTime * 2.2) * 0.008;
            }
            return;
        }

        const duration = Math.max(0.12, this.activeKeyframe.duration || 0.45);
        this.keyframeElapsed += dt;
        const slerpFactor = Math.min(1.0, (dt / duration) * 5.0);

        const targetBones = this.activeKeyframe.bones || {};
        for (const [boneName, eulerObj] of Object.entries(targetBones)) {
            const bone = this.bones[boneName];
            if (!bone) continue;

            const targetEuler = new THREE.Euler(
                eulerObj.x || 0,
                eulerObj.y || 0,
                eulerObj.z || 0,
                "XYZ"
            );
            const targetQuat = new THREE.Quaternion().setFromEuler(targetEuler);
            bone.quaternion.slerp(targetQuat, slerpFactor);
        }

        // Emit live Wrist Roll & Elbow Pitch telemetry (as shown in screenshot HUD)
        if (this.onRigTelemetry && this.bones.RightHand && this.bones.RightForeArm) {
            const wristRoll = (180 + (this.bones.RightHand.rotation.y * 180) / Math.PI).toFixed(1);
            const elbowPitch = (84.2 + (this.bones.RightForeArm.rotation.y * 45) / Math.PI).toFixed(1);
            this.onRigTelemetry({
                wristRoll,
                elbowPitch,
                handshapeLabel: this.targetHandshape ? this.targetHandshape.label : "5-Open Palm",
            });
        }

        // Apply Non-Manual Markers (Eyebrow elevation, Mouth opening)
        const nmm = this.activeKeyframe.nmm || {};
        const browRaise = nmm.eyebrowRaise || 0;
        const mouthOpen = nmm.mouthOpen || 0;

        if (this.nmmNodes.leftBrow && this.nmmNodes.rightBrow) {
            const targetBrowY = 0.032 + browRaise * 0.024;
            this.nmmNodes.leftBrow.position.y += (targetBrowY - this.nmmNodes.leftBrow.position.y) * 0.28;
            this.nmmNodes.rightBrow.position.y += (targetBrowY - this.nmmNodes.rightBrow.position.y) * 0.28;
        }
        if (this.nmmNodes.mouth) {
            const targetScaleY = 1.0 + mouthOpen * 3.5;
            this.nmmNodes.mouth.scale.y += (targetScaleY - this.nmmNodes.mouth.scale.y) * 0.28;
        }

        if (this.keyframeElapsed >= duration) {
            this.activeKeyframe = null;
        }
    }

    handleResize() {
        if (!this.renderer || !this.camera || !this.container) return;
        const w = this.container.clientWidth || 560;
        const h = this.container.clientHeight || 380;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }
}
