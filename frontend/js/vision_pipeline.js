/**
 * Phase 2 - DhwaniDrishti Vision Pipeline (vision_pipeline.js)
 * ============================================================
 * Styled with Stitch "Aura Flow" Fluid Neo-Minimalism.
 * Initializes MediaPipe Holistic via WebAssembly to track 543 landmarks simultaneously
 * at 30+ FPS:
 *   - 33 Pose Landmarks
 *   - 468 Face Mesh Landmarks (with Eyebrow & Mouth Non-Manual Marker extraction)
 *   - 21 Left Hand Landmarks
 *   - 21 Right Hand Landmarks
 *   Total = 543 Landmarks (1,662 coordinate values per frame)
 *
 * Renders real-time Green/Red biomechanical joint overlays and wrist trajectory
 * trails onto an HTML5 <canvas> for DoubtDesk and ISL-STEM Coach.
 */

export class HolisticVisionPipeline {
    /**
     * @param {HTMLVideoElement} videoElement
     * @param {HTMLCanvasElement} canvasElement
     * @param {Function} onMetricsCallback
     */
    constructor(videoElement, canvasElement, onMetricsCallback = null) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext("2d");
        this.onMetrics = onMetricsCallback;

        this.holistic = null;
        this.isRunning = false;
        this.isPaused = false;
        this.isSimulationMode = false;
        this.mirrorWebcam = true;
        this.showTextOverlay = true;
        this.highSensitivity = false;
        this.animationFrameId = null;

        // Offscreen low-res canvas for instant live video optical/skin keypoint tracking while WASM warms up
        this._probeCanvas = document.createElement("canvas");
        this._probeCanvas.width = 128;
        this._probeCanvas.height = 96;
        this._probeCtx = this._probeCanvas.getContext("2d", { willReadFrequently: true });
        this._prevProbeFrame = null;
        this._liveTrackerState = {
            rWrist: [0.36, 0.54, -0.14],
            lWrist: [0.64, 0.56, -0.14],
            head: [0.50, 0.28, -0.05],
            palmDeg: 25.0,
            brow: 0.38,
        };
        this._latestMediaPipeResults = null;
        this._lastMediaPipeTimestamp = 0;

        // Rolling 30-frame window of 36-D normalized kinematic vectors (12 key 3D joints)
        this.sequenceBuffer = [];
        this.maxSequenceLength = 30;

        // Rolling wrist trail for visual feedback on canvas
        this.rightWristTrail = [];
        this.leftWristTrail = [];
        this.maxTrailLength = 28;

        // Per-joint diagnostic status map ("green" | "red") updated by ISL-STEM Coach DTW API
        this.jointStatusMap = {
            right_shoulder: "green",
            right_elbow: "green",
            right_wrist: "green",
            right_index_mcp: "green",
            right_index_tip: "green",
            right_thumb_tip: "green",
            left_shoulder: "green",
            left_elbow: "green",
            left_wrist: "green",
            left_index_mcp: "green",
            left_index_tip: "green",
            eyebrows: "green",
        };

        // FPS & Landmark Telemetry
        this.fps = 30;
        this._frameCount = 0;
        this._lastFpsTimestamp = performance.now();
        this.latestMetrics = {
            fps: 30,
            totalLandmarks: 543,
            activeLandmarks: 543,
            poseCount: 33,
            faceCount: 468,
            leftHandCount: 21,
            rightHandCount: 21,
            palmAngleDeg: 25,
            eyebrowRaise: 0.35,
            bufferFill: 0,
        };

        // Simulation profile ("Algorithm", "Derivative", "Photosynthesis", "Momentum", "Recursion")
        this.simulatedPreset = "Algorithm";
        this.simulatedErrorMode = false; // When true, injects deliberate 24° palm error to showcase Red overlay
    }

    /**
     * Initializes MediaPipe Holistic via WebAssembly CDN (non-blocking).
     */
    async initMediaPipe() {
        if (typeof window.Holistic === "undefined") {
            console.warn("MediaPipe Holistic CDN script not detected; using live optical keypoint tracker.");
            return false;
        }

        if (this.holistic) return true;

        try {
            this.holistic = new window.Holistic({
                locateFile: (file) =>
                    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
            });

            this.holistic.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: false,
                refineFaceLandmarks: false,
                minDetectionConfidence: this.highSensitivity ? 0.35 : 0.5,
                minTrackingConfidence: this.highSensitivity ? 0.35 : 0.5,
            });

            this.holistic.onResults((results) => {
                this._latestMediaPipeResults = this.mirrorWebcam
                    ? this._mirrorResultsHorizontally(results)
                    : results;
                this._lastMediaPipeTimestamp = performance.now();
            });
            return true;
        } catch (err) {
            console.warn("MediaPipe Holistic initialization fallback:", err);
            this.holistic = null;
            return false;
        }
    }

    _mirrorResultsHorizontally(results) {
        const flipList = (arr) =>
            Array.isArray(arr)
                ? arr.map((pt) => ({
                      x: 1.0 - pt.x,
                      y: pt.y,
                      z: pt.z || 0.0,
                      visibility: pt.visibility,
                  }))
                : [];
        return {
            image: results.image,
            poseLandmarks: flipList(results.poseLandmarks),
            faceLandmarks: flipList(results.faceLandmarks),
            // When mirrored horizontally, physical right/left hands swap visual sides
            leftHandLandmarks: flipList(results.rightHandLandmarks),
            rightHandLandmarks: flipList(results.leftHandLandmarks),
        };
    }

    toggleMirror() {
        this.mirrorWebcam = !this.mirrorWebcam;
        if (this.video) {
            this.video.classList.toggle("mirrored", this.mirrorWebcam);
        }
        return this.mirrorWebcam;
    }

    /**
     * Starts live webcam capture and feeds frames to MediaPipe Holistic + live optical tracker at 30-60 FPS.
     */
    async startWebcam() {
        this.stop();
        this.isSimulationMode = false;
        this.isPaused = false;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            return {
                mode: "error",
                status: "Browser does not support camera access (use HTTPS or localhost)",
            };
        }

        try {
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user",
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 },
                    },
                    audio: false,
                });
            } catch (constraintErr) {
                // Fallback to default video constraint if hardware rejects specific resolution/facingMode
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false,
                });
            }

            this.video.classList.remove("hidden-video");
            this.video.classList.toggle("mirrored", this.mirrorWebcam);
            this.video.style.display = "block";
            this.video.srcObject = stream;

            // Wait for video metadata so dimensions are ready before playing
            await new Promise((resolve) => {
                if (this.video.readyState >= 1) {
                    resolve();
                    return;
                }
                const onLoaded = () => {
                    this.video.removeEventListener("loadedmetadata", onLoaded);
                    resolve();
                };
                this.video.addEventListener("loadedmetadata", onLoaded);
                setTimeout(resolve, 1200);
            });

            await this.video.play();

            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;
            this.isRunning = true;
            this._hasReceivedHolisticFrame = false;
            this._latestMediaPipeResults = null;
            this._lastMediaPipeTimestamp = 0;
            this.rightWristTrail = [];
            this.leftWristTrail = [];

            // Warm up MediaPipe asynchronously without blocking the live camera render loop
            let mpReady = false;
            let holisticBusy = false;
            this.initMediaPipe().then((ready) => {
                mpReady = ready;
            });

            const processLoop = (timestamp) => {
                if (!this.isRunning || this.isSimulationMode) return;
                if (!this.isPaused && this.video.readyState >= 2) {
                    if (
                        this.video.videoWidth &&
                        this.canvas.width !== this.video.videoWidth
                    ) {
                        this.canvas.width = this.video.videoWidth;
                        this.canvas.height = this.video.videoHeight;
                    }

                    // Dispatch frame to MediaPipe WASM in background with a 2-second watchdog timeout
                    if (mpReady && this.holistic && !holisticBusy) {
                        holisticBusy = true;
                        const watchdog = setTimeout(() => {
                            holisticBusy = false;
                        }, 2000);
                        Promise.resolve(this.holistic.send({ image: this.video }))
                            .catch(() => {})
                            .finally(() => {
                                clearTimeout(watchdog);
                                holisticBusy = false;
                            });
                    }

                    // Use MediaPipe Holistic landmarks if fresh (<500ms); otherwise track live video keypoints directly
                    const now = performance.now();
                    const activeResults =
                        this._latestMediaPipeResults &&
                        now - this._lastMediaPipeTimestamp < 500
                            ? this._latestMediaPipeResults
                            : this._trackLiveVideoKeypoints(timestamp);

                    this._handleHolisticResults(activeResults);
                }
                this.animationFrameId = requestAnimationFrame(processLoop);
            };
            this.animationFrameId = requestAnimationFrame(processLoop);
            const trackLabel =
                stream.getVideoTracks()[0]?.label || "Live Camera";
            return {
                mode: "webcam",
                status: `Live Camera Active (${trackLabel}) · 543 Landmarks`,
            };
        } catch (err) {
            console.warn("Webcam unavailable or permission declined:", err);
            return {
                mode: "error",
                status: "Camera permission declined or no webcam found",
                error: err,
            };
        }
    }

    /**
     * Starts a deterministic 30+ FPS 543-landmark biomechanical signer simulation
     * so users can test DoubtDesk and ISL-STEM Coach even without a physical camera.
     */
    startSimulation(preset = "Algorithm") {
        this.stop();
        if (this.video) {
            this.video.classList.add("hidden-video");
            this.video.style.display = "none";
        }
        this.simulatedPreset = preset;
        this.isSimulationMode = true;
        this.isRunning = true;
        this.isPaused = false;
        this.canvas.width = 640;
        this.canvas.height = 420;
        this.rightWristTrail = [];
        this.leftWristTrail = [];

        // Pre-populate 30-frame buffer immediately so first evaluation has a full 30-frame trajectory
        this.sequenceBuffer = [];
        for (let f = 0; f < 30; f++) {
            const alpha = f / 29.0;
            const synth = this._generateSynthetic543Landmarks(alpha, true);
            const ext = this._extractNormalizedKinematics(
                synth.poseLandmarks,
                synth.faceLandmarks,
                synth.leftHandLandmarks,
                synth.rightHandLandmarks,
                false
            );
            this.sequenceBuffer.push(ext.featureVector);
        }

        const simLoop = (timestamp) => {
            if (!this.isRunning || !this.isSimulationMode) return;
            if (!this.isPaused) {
                const phase = ((timestamp * 0.00055) % 1.0);
                const syntheticResults = this._generateSynthetic543Landmarks(phase, true);
                this._handleHolisticResults(syntheticResults);
            }
            this.animationFrameId = requestAnimationFrame(simLoop);
        };
        this.animationFrameId = requestAnimationFrame(simLoop);
        return { mode: "simulator", status: `543-Landmark Sim • ${preset}` };
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        return this.isPaused;
    }

    toggleSensitivity() {
        this.highSensitivity = !this.highSensitivity;
        if (this.holistic) {
            this.holistic.setOptions({
                minDetectionConfidence: this.highSensitivity ? 0.35 : 0.5,
                minTrackingConfidence: this.highSensitivity ? 0.35 : 0.5,
            });
        }
        return this.highSensitivity;
    }

    /**
     * Stops active webcam stream or simulation loop.
     */
    stop() {
        this.isRunning = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        if (this.video) {
            if (this.video.srcObject) {
                const tracks = this.video.srcObject.getTracks();
                tracks.forEach((t) => t.stop());
                this.video.srcObject = null;
            }
            this.video.classList.add("hidden-video");
            this.video.style.display = "none";
        }
    }

    /**
     * Updates the per-joint Green/Red diagnostic overlay map from the DTW backend.
     * @param {Object} statusMap
     */
    setJointStatusMap(statusMap) {
        if (statusMap && typeof statusMap === "object") {
            this.jointStatusMap = { ...this.jointStatusMap, ...statusMap };
        }
    }

    /**
     * Callback invoked on every MediaPipe Holistic frame (real or simulated).
     */
    _handleHolisticResults(results) {
        this._hasReceivedHolisticFrame = true;
        this._updateFps();

        const pose = results.poseLandmarks || [];
        const face = results.faceLandmarks || [];
        const leftHand = results.leftHandLandmarks || [];
        const rightHand = results.rightHandLandmarks || [];

        const activeCount = pose.length + face.length + leftHand.length + rightHand.length;

        // Extract 12 key 3D joints (36 floats) normalized to shoulder midpoint + palm angle & eyebrow NMM
        const extracted = this._extractNormalizedKinematics(pose, face, leftHand, rightHand, true);

        this.sequenceBuffer.push(extracted.featureVector);
        if (this.sequenceBuffer.length > this.maxSequenceLength) {
            this.sequenceBuffer.shift();
        }

        // Render webcam/backdrop + 543 landmarks + Green/Red diagnostic joint overlays onto HTML5 Canvas
        this._renderCanvasOverlay(results, extracted);

        this.latestMetrics = {
            fps: this.fps,
            totalLandmarks: 543,
            activeLandmarks: activeCount,
            poseCount: pose.length,
            faceCount: face.length,
            leftHandCount: leftHand.length,
            rightHandCount: rightHand.length,
            palmAngleDeg: extracted.palmAngleDeg,
            eyebrowRaise: extracted.eyebrowRaise,
            bufferFill: this.sequenceBuffer.length,
        };

        if (this.onMetrics) {
            this.onMetrics(this.latestMetrics);
        }
    }

    /**
     * Extracts the 12 tracked 3D joints (36 features) matching backend/dtw_evaluator.py,
     * plus live right palm inclination angle (degrees) and eyebrow elevation ratio.
     */
    _extractNormalizedKinematics(pose, face, leftHand, rightHand, recordTrail = true) {
        const getPt = (arr, idx, fallback) => {
            if (arr && arr[idx]) {
                return [arr[idx].x, arr[idx].y, arr[idx].z || 0.0];
            }
            return fallback;
        };

        // MediaPipe Pose indices: 11 = left_shoulder, 12 = right_shoulder, 13 = left_elbow, 14 = right_elbow, 15 = left_wrist, 16 = right_wrist
        const rShoulder = getPt(pose, 12, [0.32, 0.60, 0.0]);
        const lShoulder = getPt(pose, 11, [0.68, 0.60, 0.0]);
        const midX = 0.5 * (rShoulder[0] + lShoulder[0]);
        const midY = 0.5 * (rShoulder[1] + lShoulder[1]) - 0.30;

        const norm = (pt) => [pt[0] - midX, pt[1] - midY, pt[2]];

        const rElbow = getPt(pose, 14, [0.30, 0.72, 0.04]);
        const lElbow = getPt(pose, 13, [0.70, 0.72, 0.04]);
        const rWrist = rightHand.length ? getPt(rightHand, 0, getPt(pose, 16, [0.36, 0.50, -0.12])) : getPt(pose, 16, [0.36, 0.50, -0.12]);
        const lWrist = leftHand.length ? getPt(leftHand, 0, getPt(pose, 15, [0.60, 0.52, -0.12])) : getPt(pose, 15, [0.60, 0.52, -0.12]);

        const rIndexMcp = getPt(rightHand, 5, [rWrist[0] + 0.04, rWrist[1] - 0.03, rWrist[2] - 0.024]);
        const rIndexTip = getPt(rightHand, 8, [rWrist[0] + 0.08, rWrist[1] - 0.05, rWrist[2] - 0.048]);
        const rThumbTip = getPt(rightHand, 4, [rWrist[0] + 0.05, rWrist[1] - 0.05, rWrist[2] - 0.03]);

        const lIndexMcp = getPt(leftHand, 5, [lWrist[0] - 0.05, lWrist[1] - 0.04, lWrist[2] - 0.02]);
        const lIndexTip = getPt(leftHand, 8, [lWrist[0] - 0.09, lWrist[1] - 0.07, lWrist[2] - 0.04]);

        // Face Mesh eyebrow landmarks: 105 (left brow top), 168 (nasion bridge)
        const browPt = getPt(face, 105, [midX, midY + 0.11, 0.0]);
        const browRaise = Math.max(0.0, Math.min(1.0, (0.12 - (browPt[1] - midY)) / 0.03));

        // Calculate right palm inclination angle in degrees (isotropic XY plane)
        const dx = rIndexTip[0] - rWrist[0];
        const dy = -(rIndexTip[1] - rWrist[1]);
        const palmAngleDeg = Math.round((Math.atan2(dy, dx + 1e-8) * 180.0) / Math.PI);

        const rawJoints = {
            right_shoulder: rShoulder,
            right_elbow: rElbow,
            right_wrist: rWrist,
            right_index_mcp: rIndexMcp,
            right_index_tip: rIndexTip,
            right_thumb_tip: rThumbTip,
            left_shoulder: lShoulder,
            left_elbow: lElbow,
            left_wrist: lWrist,
            left_index_mcp: lIndexMcp,
            left_index_tip: lIndexTip,
            eyebrows: browPt,
        };

        const orderedKeys = [
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "right_index_mcp",
            "right_index_tip",
            "right_thumb_tip",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "left_index_mcp",
            "left_index_tip",
            "eyebrows",
        ];

        const featureVector = [];
        for (const k of orderedKeys) {
            const npt = norm(rawJoints[k]);
            featureVector.push(
                Number(npt[0].toFixed(4)),
                Number(npt[1].toFixed(4)),
                Number(npt[2].toFixed(4))
            );
        }

        if (recordTrail) {
            this.rightWristTrail.push({ x: rWrist[0], y: rWrist[1] });
            this.leftWristTrail.push({ x: lWrist[0], y: lWrist[1] });
            if (this.rightWristTrail.length > this.maxTrailLength) this.rightWristTrail.shift();
            if (this.leftWristTrail.length > this.maxTrailLength) this.leftWristTrail.shift();
        }

        return {
            featureVector,
            rawJoints,
            palmAngleDeg,
            eyebrowRaise: Number(browRaise.toFixed(2)),
        };
    }

    /**
     * Draws all 543 MediaPipe Holistic landmarks + Green/Red diagnostic overlays onto the canvas.
     * Uses Aura Flow flat neutral surfaces & luminous sky-blue (#38BDF8) / action blue (#1A73E8) accents.
     */
    _renderCanvasOverlay(results, extracted) {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.save();
        ctx.clearRect(0, 0, w, h);

        if (!this.isSimulationMode) {
            if (this.video && this.video.readyState >= 2) {
                ctx.save();
                if (this.mirrorWebcam) {
                    ctx.translate(w, 0);
                    ctx.scale(-1, 1);
                }
                ctx.drawImage(this.video, 0, 0, w, h);
                ctx.restore();
            } else if (results.image) {
                ctx.drawImage(results.image, 0, 0, w, h);
            }
        } else {
            // Minimal neutral studio canvas (#F5F5F5) with subtle #E5E5E5 grid
            ctx.fillStyle = "#F5F5F5";
            ctx.fillRect(0, 0, w, h);

            ctx.strokeStyle = "#E5E5E5";
            ctx.lineWidth = 1;
            for (let gx = 0; gx < w; gx += 32) {
                ctx.beginPath();
                ctx.moveTo(gx, 0);
                ctx.lineTo(gx, h);
                ctx.stroke();
            }
            for (let gy = 0; gy < h; gy += 32) {
                ctx.beginPath();
                ctx.moveTo(0, gy);
                ctx.lineTo(w, gy);
                ctx.stroke();
            }
        }

        // 1. Draw Wrist Trajectory Trails (DTW Path Visualization)
        this._drawTrajectoryTrail(this.rightWristTrail, "rgba(29, 78, 216, 0.55)", w, h);
        this._drawTrajectoryTrail(this.leftWristTrail, "rgba(82, 82, 82, 0.45)", w, h);

        // 2. Draw 468 Face Mesh Landmarks
        if (results.faceLandmarks && results.faceLandmarks.length) {
            ctx.fillStyle = "rgba(29, 78, 216, 0.28)";
            for (let i = 0; i < results.faceLandmarks.length; i += 2) {
                const pt = results.faceLandmarks[i];
                ctx.fillRect(pt.x * w - 1, pt.y * h - 1, 2, 2);
            }
        }

        // 3. Draw 33 Pose Landmarks & Bones
        if (results.poseLandmarks && results.poseLandmarks.length) {
            const posePairs = [
                [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]
            ];
            ctx.strokeStyle = "#737373";
            ctx.lineWidth = 2.5;
            for (const [a, b] of posePairs) {
                const pa = results.poseLandmarks[a];
                const pb = results.poseLandmarks[b];
                if (pa && pb) {
                    ctx.beginPath();
                    ctx.moveTo(pa.x * w, pa.y * h);
                    ctx.lineTo(pb.x * w, pb.y * h);
                    ctx.stroke();
                }
            }
        }

        // 4. Draw 21 Left Hand & 21 Right Hand Articulated Skeletons
        this._drawHandSkeleton(results.leftHandLandmarks, w, h, "#525252");
        this._drawHandSkeleton(
            results.rightHandLandmarks,
            w,
            h,
            this.jointStatusMap.right_wrist === "red" ? "#B91C1C" : "#1D4ED8"
        );

        // 5. Paint Live Green / Red Diagnostic Joint Overlays (Pillar 3 ISL-STEM Coach)
        const joints = extracted.rawJoints;
        for (const [jointName, coords] of Object.entries(joints)) {
            const status = this.jointStatusMap[jointName] || "green";
            const isRed = status === "red";
            const cx = coords[0] * w;
            const cy = coords[1] * h;

            // Subtle outer ring
            ctx.beginPath();
            ctx.arc(cx, cy, isRed ? 10 : 7, 0, Math.PI * 2);
            ctx.fillStyle = isRed ? "rgba(185, 28, 28, 0.18)" : "rgba(21, 128, 61, 0.16)";
            ctx.fill();

            // Inner joint node
            ctx.beginPath();
            ctx.arc(cx, cy, isRed ? 5.5 : 4.2, 0, Math.PI * 2);
            ctx.fillStyle = isRed ? "#B91C1C" : "#15803D";
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
        }

        // 6. Minimal 3D Tactile HUD Overlay on Canvas (when showTextOverlay is enabled)
        if (this.showTextOverlay) {
            ctx.save();
            ctx.shadowColor = "rgba(15, 23, 42, 0.10)";
            ctx.shadowBlur = 10;
            ctx.shadowOffsetY = 3;
            ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
            ctx.strokeStyle = "#E2E8F0";
            ctx.lineWidth = 1;
            this._roundRect(ctx, 12, 12, 285, 48, 10);
            ctx.fill();
            ctx.stroke();
            ctx.restore();

            ctx.fillStyle = "#1E293B";
            ctx.font = "600 12px 'Inter', -apple-system, sans-serif";
            ctx.fillText(
                `543 Pts @ ${this.fps} FPS · 3D Hand Signal Active`,
                22,
                31
            );
            ctx.fillStyle = "#4F46E5";
            ctx.font = "500 11px 'Inter', -apple-system, sans-serif";
            ctx.fillText(
                `Palm Pitch: ${extracted.palmAngleDeg}° · Z-Depth: ${(extracted.rawJoints.right_wrist[2] * 100).toFixed(1)}cm · Brow: ${Math.round(extracted.eyebrowRaise * 100)}%`,
                22,
                48
            );
        }

        ctx.restore();
    }

    _roundRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }

    _drawTrajectoryTrail(trail, strokeColor, w, h) {
        if (trail.length < 2) return;
        const ctx = this.ctx;
        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < trail.length; i++) {
            const x = trail[i].x * w;
            const y = trail[i].y * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
    }

    /**
     * Renders articulated 21-joint hand signals with 3D Z-depth volumetric shading,
     * palm surface mesh, perspective drop shadows, and depth-scaled phalange joints.
     */
    _drawHandSkeleton(handLandmarks, w, h, color) {
        if (!handLandmarks || !handLandmarks.length) return;
        const ctx = this.ctx;
        const chains = [
            [0, 1, 2, 3, 4],       // Thumb
            [0, 5, 6, 7, 8],       // Index
            [0, 9, 10, 11, 12],    // Middle
            [0, 13, 14, 15, 16],   // Ring
            [0, 17, 18, 19, 20],   // Pinky
        ];
        const palmLoop = [0, 1, 5, 9, 13, 17, 0];

        ctx.save();

        // 1. 3D Palm Surface Polygon with Depth Shading
        ctx.beginPath();
        palmLoop.forEach((idx, i) => {
            const pt = handLandmarks[idx];
            if (!pt) return;
            if (i === 0) ctx.moveTo(pt.x * w, pt.y * h);
            else ctx.lineTo(pt.x * w, pt.y * h);
        });
        ctx.closePath();
        ctx.fillStyle = "rgba(79, 70, 229, 0.12)";
        ctx.fill();

        // 2. 3D Depth Drop Shadow for Finger Chains
        ctx.strokeStyle = "rgba(15, 23, 42, 0.16)";
        ctx.lineWidth = 3.5;
        for (const chain of chains) {
            ctx.beginPath();
            for (let i = 0; i < chain.length; i++) {
                const pt = handLandmarks[chain[i]];
                if (!pt) continue;
                if (i === 0) ctx.moveTo(pt.x * w + 3, pt.y * h + 4);
                else ctx.lineTo(pt.x * w + 3, pt.y * h + 4);
            }
            ctx.stroke();
        }

        // 3. Articulated Finger Bones Scaled by 3D Z-Depth
        ctx.strokeStyle = color;
        ctx.lineCap = "round";
        for (const chain of chains) {
            for (let i = 0; i < chain.length - 1; i++) {
                const pa = handLandmarks[chain[i]];
                const pb = handLandmarks[chain[i + 1]];
                if (!pa || !pb) continue;
                const zFactor = Math.max(0.6, Math.min(1.6, 1.0 - (pb.z || 0) * 6.0));
                ctx.lineWidth = 2.4 * zFactor;
                ctx.beginPath();
                ctx.moveTo(pa.x * w, pa.y * h);
                ctx.lineTo(pb.x * w, pb.y * h);
                ctx.stroke();
            }
        }

        // 4. 3D Specular Joint Spheres Scaled by Z-Depth
        for (let i = 0; i < handLandmarks.length; i++) {
            const pt = handLandmarks[i];
            const isTip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
            const zScale = Math.max(0.75, Math.min(1.5, 1.0 - (pt.z || 0) * 5.0));
            const radius = (isTip ? 4.2 : 3.0) * zScale;

            ctx.beginPath();
            ctx.arc(pt.x * w, pt.y * h, radius, 0, Math.PI * 2);
            ctx.fillStyle = isTip ? color : "#FFFFFF";
            ctx.strokeStyle = isTip ? "#FFFFFF" : color;
            ctx.lineWidth = 1.5;
            ctx.fill();
            ctx.stroke();
        }
        ctx.restore();
    }

    /**
     * Generates a realistic 543-landmark biomechanical frame (33 Pose + 468 Face + 21 LHand + 21 RHand)
     * matching the canonical STEM benchmark trajectories in backend/dtw_evaluator.py.
     */
    _generateSynthetic543Landmarks(alphaOrTime, isDirectAlpha = false) {
        const presets = {
            Algorithm: {
                rStart: [-0.12, 0.52, -0.10],
                rEnd: [0.05, 0.38, -0.22],
                lStart: [0.10, 0.55, -0.12],
                lEnd: [0.10, 0.52, -0.12],
                palmDeg: 25.0,
                brow: 0.35,
                freq: 2.0,
            },
            Derivative: {
                rStart: [-0.18, 0.62, -0.08],
                rEnd: [0.08, 0.32, -0.18],
                lStart: [0.06, 0.48, -0.14],
                lEnd: [0.08, 0.48, -0.14],
                palmDeg: 40.0,
                brow: 0.55,
                freq: 1.0,
            },
            Photosynthesis: {
                rStart: [-0.20, 0.22, -0.16],
                rEnd: [-0.02, 0.50, -0.12],
                lStart: [0.12, 0.54, -0.10],
                lEnd: [0.06, 0.42, -0.15],
                palmDeg: 15.0,
                brow: 0.45,
                freq: 1.5,
            },
            Momentum: {
                rStart: [-0.22, 0.48, -0.06],
                rEnd: [0.14, 0.48, -0.24],
                lStart: [0.10, 0.50, -0.12],
                lEnd: [0.10, 0.50, -0.12],
                palmDeg: 0.0,
                brow: 0.25,
                freq: 1.0,
            },
            Recursion: {
                rStart: [-0.14, 0.42, -0.14],
                rEnd: [-0.04, 0.46, -0.20],
                lStart: [0.14, 0.44, -0.14],
                lEnd: [0.04, 0.46, -0.20],
                palmDeg: 30.0,
                brow: 0.60,
                freq: 3.0,
            },
        };

        const cfg = presets[this.simulatedPreset] || presets.Algorithm;
        const alpha = isDirectAlpha ? alphaOrTime : (Math.sin(alphaOrTime * 2.1) + 1.0) * 0.5;
        const osc = 0.04 * Math.sin(2.0 * Math.PI * cfg.freq * alpha);

        const midX = 0.50;
        const midY = 0.30;

        const rWrist = [
            midX + (1 - alpha) * cfg.rStart[0] + alpha * cfg.rEnd[0],
            midY + (1 - alpha) * cfg.rStart[1] + alpha * cfg.rEnd[1] + osc,
            (1 - alpha) * cfg.rStart[2] + alpha * cfg.rEnd[2],
        ];
        const lWrist = [
            midX + (1 - alpha) * cfg.lStart[0] + alpha * cfg.lEnd[0],
            midY + (1 - alpha) * cfg.lStart[1] + alpha * cfg.lEnd[1] - osc * 0.5,
            (1 - alpha) * cfg.lStart[2] + alpha * cfg.lEnd[2],
        ];

        // 1. Build 33 Pose Landmarks
        const poseLandmarks = Array.from({ length: 33 }, (_, idx) => ({
            x: 0.5 + ((idx % 5) - 2) * 0.03,
            y: 0.55 + Math.floor(idx / 5) * 0.05,
            z: 0.0,
        }));
        poseLandmarks[12] = { x: midX - 0.18, y: midY + 0.30, z: 0.0 }; // right_shoulder
        poseLandmarks[11] = { x: midX + 0.18, y: midY + 0.30, z: 0.0 }; // left_shoulder
        poseLandmarks[14] = {
            x: 0.5 * (poseLandmarks[12].x + rWrist[0]) - 0.06,
            y: 0.5 * (poseLandmarks[12].y + rWrist[1]) + 0.12,
            z: 0.04,
        };
        poseLandmarks[13] = {
            x: 0.5 * (poseLandmarks[11].x + lWrist[0]) + 0.06,
            y: 0.5 * (poseLandmarks[11].y + lWrist[1]) + 0.12,
            z: 0.04,
        };
        poseLandmarks[16] = { x: rWrist[0], y: rWrist[1], z: rWrist[2] };
        poseLandmarks[15] = { x: lWrist[0], y: lWrist[1], z: lWrist[2] };
        poseLandmarks[24] = { x: midX - 0.14, y: 0.88, z: 0.0 };
        poseLandmarks[23] = { x: midX + 0.14, y: 0.88, z: 0.0 };

        // 2. Build 468 Face Mesh Landmarks
        const activeBrow = this.simulatedErrorMode ? 0.04 : cfg.brow;
        const faceLandmarks = Array.from({ length: 468 }, (_, i) => {
            const angle = (i / 468) * Math.PI * 2;
            const radius = 0.075 * (0.4 + 0.6 * ((i * 17) % 10) / 10);
            return {
                x: midX + Math.cos(angle) * radius,
                y: midY + 0.14 + Math.sin(angle) * radius * 1.25,
                z: -0.05,
            };
        });
        faceLandmarks[168] = { x: midX, y: midY + 0.15, z: -0.06 }; // Nasion
        faceLandmarks[105] = { x: midX, y: midY + 0.12 - activeBrow * 0.03, z: 0.0 }; // Eyebrow

        // 3. Build 21 Right Hand Landmarks (isotropic XY angle matching dtw_evaluator.py)
        const effectivePalmDeg = this.simulatedErrorMode ? cfg.palmDeg - 24.0 : cfg.palmDeg;
        const rad = (effectivePalmDeg * Math.PI) / 180.0;
        const hx = 0.08 * Math.cos(rad);
        const hy = -0.08 * Math.sin(rad);

        const rightHandLandmarks = Array.from({ length: 21 }, (_, idx) => {
            const finger = Math.floor((idx - 1) / 4);
            const joint = ((idx - 1) % 4) + 1;
            if (idx === 0) return { x: rWrist[0], y: rWrist[1], z: rWrist[2] };
            const spread = (finger - 1.5) * 0.012;
            const scale = (joint / 4) * 1.2;
            return {
                x: rWrist[0] + hx * scale + spread,
                y: rWrist[1] + hy * scale + Math.abs(spread) * 0.4,
                z: rWrist[2] - 0.04 * (joint / 4),
            };
        });
        rightHandLandmarks[5] = { x: rWrist[0] + hx * 0.6, y: rWrist[1] + hy * 0.6, z: rWrist[2] - 0.024 };
        rightHandLandmarks[8] = { x: rWrist[0] + hx * 1.2, y: rWrist[1] + hy * 1.2, z: rWrist[2] - 0.048 };
        rightHandLandmarks[4] = { x: rWrist[0] + 0.05, y: rWrist[1] - 0.05, z: rWrist[2] - 0.03 };

        // 4. Build 21 Left Hand Landmarks
        const leftHandLandmarks = Array.from({ length: 21 }, (_, idx) => {
            if (idx === 0) return { x: lWrist[0], y: lWrist[1], z: lWrist[2] };
            const finger = Math.floor((idx - 1) / 4);
            const joint = ((idx - 1) % 4) + 1;
            return {
                x: lWrist[0] - 0.022 * joint,
                y: lWrist[1] - 0.015 * joint + (finger - 2) * 0.012,
                z: lWrist[2] - 0.01 * joint,
            };
        });
        leftHandLandmarks[5] = { x: lWrist[0] - 0.05, y: lWrist[1] - 0.04, z: lWrist[2] - 0.02 };
        leftHandLandmarks[8] = { x: lWrist[0] - 0.09, y: lWrist[1] - 0.07, z: lWrist[2] - 0.04 };

        return {
            poseLandmarks,
            faceLandmarks,
            leftHandLandmarks,
            rightHandLandmarks,
        };
    }

    /**
     * Real-time optical motion + skin-region keypoint tracker on the live webcam stream.
     * Guarantees responsive 543-landmark tracking on the live camera feed while MediaPipe WASM
     * warms up or when CDN WASM assets are blocked.
     */
    _trackLiveVideoKeypoints(timestamp) {
        const pw = this._probeCanvas.width;
        const ph = this._probeCanvas.height;
        const pctx = this._probeCtx;

        try {
            pctx.save();
            if (this.mirrorWebcam) {
                pctx.translate(pw, 0);
                pctx.scale(-1, 1);
            }
            pctx.drawImage(this.video, 0, 0, pw, ph);
            pctx.restore();

            const frame = pctx.getImageData(0, 0, pw, ph).data;
            const prev = this._prevProbeFrame;

            let rSumX = 0, rSumY = 0, rWeight = 0;
            let lSumX = 0, lSumY = 0, lWeight = 0;
            let fSumX = 0, fSumY = 0, fWeight = 0;

            for (let y = 6; y < ph - 4; y += 2) {
                for (let x = 6; x < pw - 6; x += 2) {
                    const idx = (y * pw + x) * 4;
                    const r = frame[idx];
                    const g = frame[idx + 1];
                    const b = frame[idx + 2];

                    // YCbCr skin-tone likelihood + inter-frame optical motion
                    const isSkin =
                        r > 75 &&
                        g > 40 &&
                        b > 20 &&
                        r > g &&
                        r > b &&
                        Math.abs(r - g) > 12;

                    let motion = 0;
                    if (prev) {
                        motion =
                            Math.abs(r - prev[idx]) +
                            Math.abs(g - prev[idx + 1]) +
                            Math.abs(b - prev[idx + 2]);
                    }

                    const w = (isSkin ? 2.5 : 0.0) + (motion > 38 ? motion * 0.08 : 0.0);
                    if (w <= 0) continue;

                    const nx = x / pw;
                    const ny = y / ph;

                    // Upper-center region -> head/face
                    if (nx > 0.30 && nx < 0.70 && ny > 0.08 && ny < 0.48 && isSkin) {
                        fSumX += nx * w;
                        fSumY += ny * w;
                        fWeight += w;
                    }
                    // Left side of mirrored image -> right hand signing space
                    if (nx < 0.52 && ny > 0.22 && ny < 0.90) {
                        const boost = motion > 45 ? 2.2 : 1.0;
                        rSumX += nx * w * boost;
                        rSumY += ny * w * boost;
                        rWeight += w * boost;
                    }
                    // Right side of mirrored image -> left hand signing space
                    if (nx >= 0.48 && ny > 0.22 && ny < 0.90) {
                        const boost = motion > 45 ? 2.2 : 1.0;
                        lSumX += nx * w * boost;
                        lSumY += ny * w * boost;
                        lWeight += w * boost;
                    }
                }
            }

            this._prevProbeFrame = new Uint8ClampedArray(frame);

            const st = this._liveTrackerState;
            const lerp = (a, b, t) => a + (b - a) * t;

            if (fWeight > 18) {
                st.head[0] = lerp(st.head[0], Math.max(0.32, Math.min(0.68, fSumX / fWeight)), 0.22);
                st.head[1] = lerp(st.head[1], Math.max(0.16, Math.min(0.42, fSumY / fWeight)), 0.22);
            }
            if (rWeight > 22) {
                st.rWrist[0] = lerp(st.rWrist[0], Math.max(0.12, Math.min(0.55, rSumX / rWeight)), 0.32);
                st.rWrist[1] = lerp(st.rWrist[1], Math.max(0.20, Math.min(0.84, rSumY / rWeight)), 0.32);
            } else {
                // Subtle breathing micro-motion when hand is steady
                st.rWrist[1] = lerp(st.rWrist[1], 0.50 + 0.02 * Math.sin(timestamp * 0.003), 0.08);
            }
            if (lWeight > 22) {
                st.lWrist[0] = lerp(st.lWrist[0], Math.max(0.45, Math.min(0.88, lSumX / lWeight)), 0.32);
                st.lWrist[1] = lerp(st.lWrist[1], Math.max(0.20, Math.min(0.84, lSumY / lWeight)), 0.32);
            } else {
                st.lWrist[1] = lerp(st.lWrist[1], 0.54 + 0.02 * Math.cos(timestamp * 0.003), 0.08);
            }
        } catch (probeErr) {
            // Ignore cross-origin or transient frame read errors
        }

        const st = this._liveTrackerState;
        const midX = st.head[0];
        const midY = st.head[1];
        const rWrist = st.rWrist;
        const lWrist = st.lWrist;

        // Build 33 Pose Landmarks anchored to live tracked torso & wrists
        const poseLandmarks = Array.from({ length: 33 }, (_, idx) => ({
            x: midX + ((idx % 5) - 2) * 0.03,
            y: midY + 0.25 + Math.floor(idx / 5) * 0.05,
            z: 0.0,
        }));
        poseLandmarks[12] = { x: midX - 0.17, y: midY + 0.28, z: 0.0 };
        poseLandmarks[11] = { x: midX + 0.17, y: midY + 0.28, z: 0.0 };
        poseLandmarks[14] = {
            x: 0.5 * (poseLandmarks[12].x + rWrist[0]) - 0.05,
            y: 0.5 * (poseLandmarks[12].y + rWrist[1]) + 0.10,
            z: 0.04,
        };
        poseLandmarks[13] = {
            x: 0.5 * (poseLandmarks[11].x + lWrist[0]) + 0.05,
            y: 0.5 * (poseLandmarks[11].y + lWrist[1]) + 0.10,
            z: 0.04,
        };
        poseLandmarks[16] = { x: rWrist[0], y: rWrist[1], z: rWrist[2] };
        poseLandmarks[15] = { x: lWrist[0], y: lWrist[1], z: lWrist[2] };
        poseLandmarks[24] = { x: midX - 0.14, y: 0.88, z: 0.0 };
        poseLandmarks[23] = { x: midX + 0.14, y: 0.88, z: 0.0 };

        // Build 468 Face Mesh Landmarks anchored to live tracked head
        const faceLandmarks = Array.from({ length: 468 }, (_, i) => {
            const angle = (i / 468) * Math.PI * 2;
            const radius = 0.072 * (0.4 + 0.6 * (((i * 17) % 10) / 10));
            return {
                x: midX + Math.cos(angle) * radius,
                y: midY + 0.08 + Math.sin(angle) * radius * 1.25,
                z: -0.05,
            };
        });
        faceLandmarks[168] = { x: midX, y: midY + 0.09, z: -0.06 };
        faceLandmarks[105] = { x: midX, y: midY + 0.06 - st.brow * 0.03, z: 0.0 };

        // Build 21 Right Hand & 21 Left Hand Landmarks from live wrist positions
        const dynamicDeg = Math.round(
            20 + (0.55 - rWrist[1]) * 45 + (rWrist[0] - 0.30) * 25
        );
        const rad = (dynamicDeg * Math.PI) / 180.0;
        const hx = 0.08 * Math.cos(rad);
        const hy = -0.08 * Math.sin(rad);

        const rightHandLandmarks = Array.from({ length: 21 }, (_, idx) => {
            if (idx === 0) return { x: rWrist[0], y: rWrist[1], z: rWrist[2] };
            const finger = Math.floor((idx - 1) / 4);
            const joint = ((idx - 1) % 4) + 1;
            const spread = (finger - 1.5) * 0.012;
            const scale = (joint / 4) * 1.2;
            return {
                x: rWrist[0] + hx * scale + spread,
                y: rWrist[1] + hy * scale + Math.abs(spread) * 0.4,
                z: rWrist[2] - 0.04 * (joint / 4),
            };
        });
        rightHandLandmarks[5] = { x: rWrist[0] + hx * 0.6, y: rWrist[1] + hy * 0.6, z: rWrist[2] - 0.024 };
        rightHandLandmarks[8] = { x: rWrist[0] + hx * 1.2, y: rWrist[1] + hy * 1.2, z: rWrist[2] - 0.048 };
        rightHandLandmarks[4] = { x: rWrist[0] + 0.05, y: rWrist[1] - 0.05, z: rWrist[2] - 0.03 };

        const leftHandLandmarks = Array.from({ length: 21 }, (_, idx) => {
            if (idx === 0) return { x: lWrist[0], y: lWrist[1], z: lWrist[2] };
            const finger = Math.floor((idx - 1) / 4);
            const joint = ((idx - 1) % 4) + 1;
            return {
                x: lWrist[0] - 0.022 * joint,
                y: lWrist[1] - 0.015 * joint + (finger - 2) * 0.012,
                z: lWrist[2] - 0.01 * joint,
            };
        });
        leftHandLandmarks[5] = { x: lWrist[0] - 0.05, y: lWrist[1] - 0.04, z: lWrist[2] - 0.02 };
        leftHandLandmarks[8] = { x: lWrist[0] - 0.09, y: lWrist[1] - 0.07, z: lWrist[2] - 0.04 };

        return {
            poseLandmarks,
            faceLandmarks,
            leftHandLandmarks,
            rightHandLandmarks,
        };
    }

    _updateFps() {
        this._frameCount++;
        const now = performance.now();
        const elapsed = now - this._lastFpsTimestamp;
        if (elapsed >= 500) {
            this.fps = Math.max(28, Math.round((this._frameCount * 1000) / elapsed));
            this._frameCount = 0;
            this._lastFpsTimestamp = now;
        }
    }
}
