/**
 * DhwaniDrishti Main Application Orchestrator (app.js)
 * ====================================================
 * Minimal, calm, and accessible UI controller coordinating the four modes:
 *   1. LectureCast (Teacher -> Student): Web Speech API + FastAPI spaCy SOV Engine + Three.js 3D Avatar
 *   2. DoubtDesk (Student -> Teacher): MediaPipe Holistic 543-Landmark Stream + 1D-CNN/LSTM + Web Speech Synthesis
 *   3. ISL-STEM Coach (Interactive Tutor): Gamified DTW + Vector Cosine Distance + Green/Red Joint Overlays
 *   4. Live Spatial Viewfinders / 3D ISL Signer
 */

import { HolisticVisionPipeline } from "./vision_pipeline.js";
import { ISLAvatarRenderer } from "./avatar_renderer.js";

const API_BASE = window.location.origin.startsWith("http")
    ? window.location.origin
    : "http://127.0.0.1:8000";

class DhwaniDrishtiApp {
    constructor() {
        this.avatarRenderer = null;
        this.visionPipeline = null;
        this.recognition = null;
        this.isListeningSpeech = false;
        this.cameraEnabled = false;

        this.stemTerms = [];
        this.selectedStemTerm = "Algorithm";
        this.autoEvalTimer = null;
        this.xpScore = 120;
        this.streakCount = 3;

        this._initDOM();
    }

    async _initDOM() {
        // Initialize 3D ISL Avatar Renderer (Phase 4)
        const avatarContainer = document.getElementById("avatar-viewport");
        const activeGlossBadge = document.getElementById("active-avatar-gloss");
        const rigAngleReadout = document.getElementById("rig-angle-readout");
        if (avatarContainer) {
            this.avatarRenderer = new ISLAvatarRenderer(
                avatarContainer,
                (gloss) => {
                    if (activeGlossBadge) activeGlossBadge.textContent = `SIGNING: ${gloss}`;
                    this._highlightActiveGlossPill(gloss);
                },
                (telemetry) => {
                    if (rigAngleReadout) {
                        rigAngleReadout.innerHTML = `Wrist Roll: <strong>${telemetry.wristRoll}°</strong> &nbsp; Elbow Pitch: <strong>${telemetry.elbowPitch}°</strong>`;
                    }
                }
            );
        }

        // Initialize MediaPipe Holistic Vision Pipeline (Phase 2)
        const videoEl = document.getElementById("webcam-video");
        const canvasEl = document.getElementById("holistic-canvas");
        if (videoEl && canvasEl) {
            this.visionPipeline = new HolisticVisionPipeline(videoEl, canvasEl, (metrics) => {
                this._updateVisionTelemetryUI(metrics);
            });
            this.visionPipeline.startSimulation("Algorithm");
        }

        this._bindNavigationAndEvents();
        await this._checkBackendHealth();
        await this._loadStemTerms();

        // Trigger initial LectureCast parse demonstration on startup
        await this.runLectureCastParse(
            "Today the teacher explains the derivative algorithm to the students?",
            "en-IN"
        );
    }

    _highlightActiveGlossPill(activeGloss) {
        const pills = document.querySelectorAll("#isl-gloss-output .gloss-pill");
        pills.forEach((pill) => {
            if (pill.textContent === activeGloss) {
                pill.classList.add("active-signing");
            } else if (!pill.textContent.startsWith("IX-")) {
                pill.classList.remove("active-signing");
            }
        });
    }

    _bindNavigationAndEvents() {
        // Three-Line Hamburger Menu Button & Popover
        const btnMenuToggle = document.getElementById("btn-menu-toggle");
        const menuPopover = document.getElementById("workspace-menu-popover");
        const menuWrapper = document.getElementById("menu-dropdown-wrapper");

        const closeMenu = () => {
            if (!menuPopover || !btnMenuToggle) return;
            menuPopover.classList.add("hidden-menu");
            btnMenuToggle.setAttribute("aria-expanded", "false");
        };

        if (btnMenuToggle && menuPopover) {
            btnMenuToggle.addEventListener("click", (e) => {
                e.stopPropagation();
                const isHidden = menuPopover.classList.toggle("hidden-menu");
                btnMenuToggle.setAttribute("aria-expanded", String(!isHidden));
            });

            document.addEventListener("click", (e) => {
                if (menuWrapper && !menuWrapper.contains(e.target)) {
                    closeMenu();
                }
            });

            document.addEventListener("keydown", (e) => {
                if (e.key === "Escape") closeMenu();
            });
        }

        // Workspace Mode Menu Items (4 Main Modes)
        document.querySelectorAll(".mode-tab-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".mode-tab-btn").forEach((b) => {
                    b.classList.remove("active");
                    b.setAttribute("aria-selected", "false");
                });
                btn.classList.add("active");
                btn.setAttribute("aria-selected", "true");
                const view = btn.getAttribute("data-view");
                const title = btn.getAttribute("data-title") || btn.textContent.trim();
                const headerTitle = document.getElementById("current-mode-title");
                if (headerTitle) headerTitle.textContent = title;
                this._switchModeView(view);
                closeMenu();
            });
        });

        // Header Camera Enable/Disable Button
        const btnHeaderCamera = document.getElementById("btn-header-camera");
        if (btnHeaderCamera) {
            btnHeaderCamera.addEventListener("click", () => this.toggleCameraEnabled());
        }

        // Header Mic Enable/Disable Button
        const btnHeaderMic = document.getElementById("btn-header-mic");
        if (btnHeaderMic) {
            btnHeaderMic.addEventListener("click", () => this.toggleSpeechRecognition());
        }

        // 3D Avatar Persona Switcher (Aarav, Prof. Meera, Sheru Mascot)
        document.querySelectorAll(".persona-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".persona-btn").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                const persona = btn.getAttribute("data-persona");
                if (this.avatarRenderer) {
                    this.avatarRenderer.setPersona(persona);
                }
            });
        });

        // Interactive 3D Hand Signals (Mudras / Handshapes) Strip
        document.querySelectorAll(".hand-signal-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".hand-signal-btn").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                const handshape = btn.getAttribute("data-handshape");
                if (this.avatarRenderer) {
                    this.avatarRenderer.triggerHandSignal(handshape);
                }
            });
        });

        // LectureCast controls
        const btnParse = document.getElementById("btn-translate-isl");
        if (btnParse) {
            btnParse.addEventListener("click", () => {
                const input = document.getElementById("lecture-input").value;
                const lang = document.getElementById("lecture-lang").value;
                this.runLectureCastParse(input, lang);
            });
        }

        const lectureInput = document.getElementById("lecture-input");
        if (lectureInput) {
            lectureInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    const lang = document.getElementById("lecture-lang").value;
                    this.runLectureCastParse(lectureInput.value, lang);
                }
            });
        }

        const btnMic = document.getElementById("btn-speech-mic");
        if (btnMic) {
            btnMic.addEventListener("click", () => this.toggleSpeechRecognition());
        }

        // Preset STEM lecture sentence pills
        document.querySelectorAll(".preset-sentence-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const text = btn.getAttribute("data-text");
                const lang = btn.getAttribute("data-lang") || "en-IN";
                document.getElementById("lecture-input").value = text;
                document.getElementById("lecture-lang").value = lang;
                this.runLectureCastParse(text, lang);
            });
        });

        // External .glb Avatar Loader
        const btnLoadGlb = document.getElementById("btn-load-glb");
        if (btnLoadGlb) {
            btnLoadGlb.addEventListener("click", async () => {
                const url = document.getElementById("glb-url-input").value.trim();
                const statusEl = document.getElementById("glb-status");
                if (!url) {
                    statusEl.textContent = "Using built-in articulated 3D rig";
                    return;
                }
                try {
                    statusEl.textContent = "Loading .glb avatar...";
                    const res = await this.avatarRenderer.loadGLBAvatar(url);
                    statusEl.textContent = `Loaded .glb (${res.mappedBones.length} bones mapped)`;
                } catch (err) {
                    statusEl.textContent = "Built-in 3D rig active (.glb URL unreachable)";
                }
            });
        }

        // Vision Camera vs Simulator Controls
        const btnWebcam = document.getElementById("btn-start-webcam");
        if (btnWebcam) {
            btnWebcam.addEventListener("click", async () => {
                await this.toggleCameraEnabled();
            });
        }

        const btnSim = document.getElementById("btn-start-sim");
        if (btnSim) {
            btnSim.addEventListener("click", () => {
                this.cameraEnabled = false;
                this._syncCameraHeaderUI(false);
                const disabledOverlay = document.getElementById("camera-disabled-overlay");
                if (disabledOverlay) disabledOverlay.classList.add("hidden-pillar");
                const res = this.visionPipeline.startSimulation(this.selectedStemTerm);
                this._updateVisionModeBadge(res.status);
                if (btnWebcam) btnWebcam.textContent = "Use Live Webcam";
                this.runStemCoachEvaluation();
            });
        }

        // DoubtDesk Sequence Classification & Speech Synthesis controls
        const btnClassifyDoubt = document.getElementById("btn-classify-doubt");
        if (btnClassifyDoubt) {
            btnClassifyDoubt.addEventListener("click", () => this.runDoubtDeskClassification(-1));
        }

        document.querySelectorAll(".doubt-demo-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const idx = parseInt(btn.getAttribute("data-idx"), 10);
                this.runDoubtDeskClassification(idx);
            });
        });

        const btnSpeakAloud = document.getElementById("btn-speak-doubt");
        if (btnSpeakAloud) {
            btnSpeakAloud.addEventListener("click", () => {
                const text = document.getElementById("doubt-translated-text").textContent;
                const lang = document.getElementById("doubt-audio-lang").value;
                this.speakTextAloud(text, lang);
            });
        }

        // ISL-STEM Coach controls
        const termSelect = document.getElementById("stem-term-select");
        if (termSelect) {
            termSelect.addEventListener("change", (e) => {
                this.selectedStemTerm = e.target.value;
                this._renderSelectedStemTermDetails();
                if (this.visionPipeline && this.visionPipeline.isSimulationMode && this.cameraEnabled) {
                    const res = this.visionPipeline.startSimulation(this.selectedStemTerm);
                    this._updateVisionModeBadge(res.status);
                }
                this.runStemCoachEvaluation();
            });
        }

        const btnEvalDtw = document.getElementById("btn-evaluate-dtw");
        if (btnEvalDtw) {
            btnEvalDtw.addEventListener("click", () => this.runStemCoachEvaluation());
        }

        const chkInjectError = document.getElementById("chk-inject-palm-error");
        if (chkInjectError) {
            chkInjectError.addEventListener("change", (e) => {
                if (this.visionPipeline) {
                    this.visionPipeline.simulatedErrorMode = e.target.checked;
                    if (this.visionPipeline.isSimulationMode && this.cameraEnabled) {
                        this.visionPipeline.startSimulation(this.selectedStemTerm);
                    }
                }
                this.runStemCoachEvaluation();
            });
        }

        // Advanced Viewfinder Controls
        const dockCamera = document.getElementById("dock-btn-camera");
        if (dockCamera) {
            dockCamera.addEventListener("click", async () => {
                if (!this.visionPipeline) return;
                if (!this.cameraEnabled) await this.toggleCameraEnabled(true);
                if (this.visionPipeline.isSimulationMode) {
                    const res = await this.visionPipeline.startWebcam();
                    this._updateVisionModeBadge(res.status);
                } else {
                    const res = this.visionPipeline.startSimulation(this.selectedStemTerm);
                    this._updateVisionModeBadge(res.status);
                }
            });
        }

        const dockMic = document.getElementById("dock-btn-mic");
        if (dockMic) {
            dockMic.addEventListener("click", () => this.toggleSpeechRecognition());
        }

        const dockStreamToggle = document.getElementById("dock-btn-stream-toggle");
        if (dockStreamToggle) {
            dockStreamToggle.addEventListener("click", () => {
                const paused = this.visionPipeline ? this.visionPipeline.togglePause() : false;
                if (this.avatarRenderer) this.avatarRenderer.togglePause();
                const label = document.getElementById("dock-stream-label");
                const pill = document.getElementById("canvas-status-pill");
                if (paused) {
                    if (label) label.textContent = "Resume Stream";
                    if (pill) pill.textContent = "Paused";
                } else {
                    if (label) label.textContent = "Pause Stream";
                    if (pill) pill.textContent = "Active";
                }
            });
        }

        const dockSensitivity = document.getElementById("dock-btn-sensitivity");
        if (dockSensitivity) {
            dockSensitivity.addEventListener("click", () => {
                const active = this.visionPipeline ? this.visionPipeline.toggleSensitivity() : false;
                dockSensitivity.classList.toggle("active", active);
                if (active) {
                    this.runStemCoachEvaluation();
                    this.autoEvalTimer = setInterval(() => this.runStemCoachEvaluation(), 1500);
                } else if (this.autoEvalTimer) {
                    clearInterval(this.autoEvalTimer);
                    this.autoEvalTimer = null;
                }
            });
        }

        const dockOverlay = document.getElementById("dock-btn-overlay");
        if (dockOverlay) {
            dockOverlay.addEventListener("click", () => {
                if (!this.visionPipeline) return;
                this.visionPipeline.showTextOverlay = !this.visionPipeline.showTextOverlay;
                dockOverlay.classList.toggle("active", this.visionPipeline.showTextOverlay);
            });
        }
    }

    _syncCameraHeaderUI(isWebcamActive) {
        const btnHeaderCam = document.getElementById("btn-header-camera");
        const lblHeaderCam = document.getElementById("header-camera-label");
        const btnWebcam = document.getElementById("btn-start-webcam");
        if (btnHeaderCam) {
            btnHeaderCam.classList.toggle("active", isWebcamActive);
            btnHeaderCam.setAttribute("aria-pressed", String(isWebcamActive));
        }
        if (lblHeaderCam) {
            lblHeaderCam.textContent = isWebcamActive ? "Camera On" : "Camera Off";
        }
        if (btnWebcam) {
            btnWebcam.textContent = isWebcamActive ? "Disable Webcam" : "Use Live Webcam";
        }
    }

    /**
     * Enables or disables the device's physical webcam via the top header toggle button.
     */
    async toggleCameraEnabled(forceState = null) {
        const targetState = forceState !== null ? forceState : !this.cameraEnabled;
        const disabledOverlay = document.getElementById("camera-disabled-overlay");
        const statusPill = document.getElementById("canvas-status-pill");

        if (targetState) {
            if (!this.visionPipeline) return;
            this._updateVisionModeBadge("Requesting device camera access...");
            const res = await this.visionPipeline.startWebcam();
            if (res.mode === "webcam") {
                this.cameraEnabled = true;
                this._syncCameraHeaderUI(true);
                if (disabledOverlay) disabledOverlay.classList.add("hidden-pillar");
                if (statusPill) statusPill.textContent = "Webcam Active";
                this._updateVisionModeBadge(res.status);
            } else {
                this.cameraEnabled = false;
                this._syncCameraHeaderUI(false);
                this._updateVisionModeBadge(res.status || "Camera access unavailable");
                alert("Could not access the device camera. Please allow camera permissions in your browser settings.");
            }
        } else {
            this.cameraEnabled = false;
            this._syncCameraHeaderUI(false);
            if (this.visionPipeline) {
                this.visionPipeline.stop();
            }
            if (disabledOverlay) disabledOverlay.classList.remove("hidden-pillar");
            if (statusPill) statusPill.textContent = "Camera Off";
            this._updateVisionModeBadge("Device camera disabled");
        }
    }

    _switchModeView(view) {
        const workspaceMain = document.getElementById("workspace-main");
        const cardLecture = document.getElementById("card-lecturecast");
        const cardDoubt = document.getElementById("card-doubtdesk");
        const cardStem = document.getElementById("card-stem-coach");
        const detailsViewfinders = document.getElementById("details-viewfinders");

        if (workspaceMain) {
            workspaceMain.setAttribute("data-active-mode", view);
        }

        cardLecture.classList.add("hidden-pillar");
        cardDoubt.classList.add("hidden-pillar");
        cardStem.classList.add("hidden-pillar");

        if (view === "lecturecast") {
            cardLecture.classList.remove("hidden-pillar");
        } else if (view === "doubtdesk") {
            cardDoubt.classList.remove("hidden-pillar");
        } else if (view === "stemcoach") {
            cardStem.classList.remove("hidden-pillar");
            this.runStemCoachEvaluation();
        } else if (view === "viewfinders") {
            if (detailsViewfinders) detailsViewfinders.open = true;
        }

        if (this.avatarRenderer) {
            setTimeout(() => this.avatarRenderer.handleResize(), 40);
        }
    }

    _updateVisionModeBadge(statusText) {
        const badge = document.getElementById("vision-mode-badge");
        if (badge) {
            badge.textContent = statusText;
        }
    }

    async _checkBackendHealth() {
        const textEl = document.getElementById("backend-status-text");
        const pillEl = document.getElementById("backend-status-pill");
        try {
            const resp = await fetch(`${API_BASE}/api/health`);
            await resp.json();
            if (textEl) textEl.textContent = "Connected";
            if (pillEl) pillEl.classList.add("status-online");
        } catch (err) {
            if (textEl) textEl.textContent = "Offline";
            if (pillEl) pillEl.classList.remove("status-online");
        }
    }

    // -----------------------------------------------------------------------
    // PILLAR 1: LectureCast (Teacher -> Student)
    // -----------------------------------------------------------------------

    toggleSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const btnMic = document.getElementById("btn-speech-mic");
        const dockMic = document.getElementById("dock-btn-mic");
        const btnHeaderMic = document.getElementById("btn-header-mic");
        const lblHeaderMic = document.getElementById("header-mic-label");

        const setMicUI = (listening) => {
            if (btnMic) {
                btnMic.textContent = listening ? "Stop Voice" : "Voice Input";
                btnMic.classList.toggle("recording-active", listening);
            }
            if (dockMic) dockMic.classList.toggle("active", listening);
            if (btnHeaderMic) {
                btnHeaderMic.classList.toggle("mic-recording", listening);
                btnHeaderMic.setAttribute("aria-pressed", String(listening));
            }
            if (lblHeaderMic) lblHeaderMic.textContent = listening ? "Mic On" : "Mic Off";
        };

        if (!SpeechRecognition) {
            alert("Web Speech API is not supported in this browser. Use Chrome or Edge.");
            return;
        }

        if (this.isListeningSpeech && this.recognition) {
            this.recognition.stop();
            this.isListeningSpeech = false;
            setMicUI(false);
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.lang = document.getElementById("lecture-lang").value || "en-IN";
        this.recognition.continuous = true;
        this.recognition.interimResults = false;

        this.recognition.onstart = () => {
            this.isListeningSpeech = true;
            setMicUI(true);
        };

        this.recognition.onresult = (event) => {
            const lastResult = event.results[event.results.length - 1];
            if (lastResult.isFinal) {
                const transcript = lastResult[0].transcript.trim();
                document.getElementById("lecture-input").value = transcript;
                this.runLectureCastParse(transcript, this.recognition.lang);
            }
        };

        this.recognition.onerror = () => {
            this.isListeningSpeech = false;
            setMicUI(false);
        };

        this.recognition.onend = () => {
            this.isListeningSpeech = false;
            setMicUI(false);
        };

        this.recognition.start();
    }

    async runLectureCastParse(text, language = "en-IN") {
        if (!text || !text.trim()) return;
        const liveTranscript = document.getElementById("lecture-live-transcript");
        if (liveTranscript) liveTranscript.textContent = text;

        try {
            const resp = await fetch(`${API_BASE}/api/lecturecast/parse`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, language }),
            });
            const data = await resp.json();

            // Render S-O-V Gloss Pills
            const glossContainer = document.getElementById("isl-gloss-output");
            glossContainer.innerHTML = "";
            (data.isl_sov_gloss || []).forEach((gloss) => {
                const pill = document.createElement("span");
                pill.className = gloss.startsWith("IX-") ? "gloss-pill locus-pill" : "gloss-pill";
                pill.textContent = gloss;
                glossContainer.appendChild(pill);
            });

            // Render Dependency Token Breakdown
            const depContainer = document.getElementById("svo-token-breakdown");
            depContainer.innerHTML = "";
            (data.svo_tokens || []).forEach((tok) => {
                const tag = document.createElement("span");
                tag.className = "token-chip";
                tag.innerHTML = `<strong>${tok.text}</strong> <small>${tok.isl_role}</small>`;
                depContainer.appendChild(tag);
            });

            // Render NIOS 230 Sentence Type, Word-Level Compounds & Sign Families
            const niosContainer = document.getElementById("isl-nios-structures");
            if (niosContainer) {
                niosContainer.innerHTML = "";
                if (data.sentence_type) {
                    const st = document.createElement("div");
                    st.className = "marker-item spatial";
                    st.textContent = `Sentence Type: ${data.sentence_type}`;
                    niosContainer.appendChild(st);
                }
                (data.word_level_structures || []).forEach((ws) => {
                    const item = document.createElement("div");
                    item.className = "marker-item nmm";
                    const compoundStr = ws.isl_compound || (ws.components ? ws.components.join(" + ") : "");
                    item.textContent = `${ws.gloss} = ${compoundStr} (${ws.rule})`;
                    niosContainer.appendChild(item);
                });
                (data.sign_families || []).forEach((sf) => {
                    const item = document.createElement("div");
                    item.className = "marker-item spatial";
                    item.textContent = `${sf.gloss}: ${sf.family}`;
                    niosContainer.appendChild(item);
                });
            }

            // Render Spatial Markers & Non-Manual Markers
            const nmmContainer = document.getElementById("isl-nmm-output");
            nmmContainer.innerHTML = "";
            (data.spatial_markers || []).forEach((sm) => {
                const li = document.createElement("div");
                li.className = "marker-item spatial";
                li.textContent = `${sm.locus} (${sm.azimuth_deg}°): ${sm.entity} — ${sm.role}`;
                nmmContainer.appendChild(li);
            });
            (data.non_manual_markers || []).forEach((nmm) => {
                const li = document.createElement("div");
                li.className = "marker-item nmm";
                li.textContent = nmm;
                nmmContainer.appendChild(li);
            });

            // Trigger 3D Avatar Bone Rotation Keyframes
            if (this.avatarRenderer && data.avatar_keyframes) {
                this.avatarRenderer.enqueueKeyframes(data.avatar_keyframes);
            }
        } catch (err) {
            console.error("LectureCast parse error:", err);
        }
    }

    // -----------------------------------------------------------------------
    // PILLAR 2: DoubtDesk (Student -> Teacher)
    // -----------------------------------------------------------------------

    async runDoubtDeskClassification(hintIndex = -1) {
        const sequence =
            this.visionPipeline && this.visionPipeline.sequenceBuffer.length > 0
                ? this.visionPipeline.sequenceBuffer
                : [[0.0, 0.1, 0.0]];

        const targetLanguage = document.getElementById("doubt-audio-lang").value || "en-IN";

        try {
            const resp = await fetch(`${API_BASE}/api/doubtdesk/classify`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    sequence,
                    target_language: targetLanguage,
                    hint_index: hintIndex,
                }),
            });
            const data = await resp.json();

            document.getElementById("doubt-gloss-seq").textContent = data.isl_gloss;
            document.getElementById("doubt-translated-text").textContent = data.translated_text;
            document.getElementById("doubt-confidence").textContent =
                `${Math.round(data.confidence * 100)}% confidence · ${data.frames_analyzed} frames`;

            // Render trilingual preview cards
            if (data.translations) {
                document.getElementById("doubt-tr-en").textContent = data.translations.en;
                document.getElementById("doubt-tr-te").textContent = data.translations.te;
                document.getElementById("doubt-tr-hi").textContent = data.translations.hi;
            }

            // Synthesize Classroom Speaker Audio via Web Speech Synthesis API
            this.speakTextAloud(data.translated_text, targetLanguage);
        } catch (err) {
            console.error("DoubtDesk classification error:", err);
        }
    }

    speakTextAloud(text, lang = "en-IN") {
        if (!("speechSynthesis" in window) || !text) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang;
        utterance.rate = 0.96;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }

    // -----------------------------------------------------------------------
    // PILLAR 3: ISL-STEM Coach (Interactive Tutor - DTW & Cosine Distance)
    // -----------------------------------------------------------------------

    async _loadStemTerms() {
        try {
            const resp = await fetch(`${API_BASE}/api/stem-coach/terms`);
            const data = await resp.json();
            this.stemTerms = data.terms || [];

            const select = document.getElementById("stem-term-select");
            if (select && this.stemTerms.length) {
                select.innerHTML = "";
                this.stemTerms.forEach((t) => {
                    const opt = document.createElement("option");
                    opt.value = t.id;
                    opt.textContent = `${t.title} (${t.difficulty})`;
                    select.appendChild(opt);
                });
                this.selectedStemTerm = this.stemTerms[0].id;
                this._renderSelectedStemTermDetails();
                await this.runStemCoachEvaluation();
            }
        } catch (err) {
            console.error("Failed to load STEM benchmarks:", err);
        }
    }

    _renderSelectedStemTermDetails() {
        const term = this.stemTerms.find((t) => t.id === this.selectedStemTerm);
        if (!term) return;
        document.getElementById("stem-term-category").textContent = term.category;
        document.getElementById("stem-term-desc").textContent = term.isl_description;
        document.getElementById("stem-target-angle").textContent =
            `Target Palm: ${term.target_palm_angle_deg}° · Eyebrow NMM: ${Math.round(term.target_eyebrow_raise * 100)}%`;
    }

    async runStemCoachEvaluation() {
        if (!this.visionPipeline || this.visionPipeline.sequenceBuffer.length === 0) return;

        const studentSeq = this.visionPipeline.sequenceBuffer;
        const palmDeg = this.visionPipeline.latestMetrics.palmAngleDeg;
        const browRaise = this.visionPipeline.latestMetrics.eyebrowRaise;

        try {
            const resp = await fetch(`${API_BASE}/api/stem-coach/evaluate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    term_id: this.selectedStemTerm,
                    student_sequence: studentSeq,
                    measured_palm_angle_deg: palmDeg,
                    measured_eyebrow_raise: browRaise,
                }),
            });
            const result = await resp.json();

            // Paint Green/Red joint overlays onto live MediaPipe Holistic Canvas
            this.visionPipeline.setJointStatusMap(result.joint_status);

            // Update DTW & Vector Cosine Telemetry UI
            document.getElementById("metric-dtw-dist").textContent = result.dtw_distance.toFixed(4);
            document.getElementById("metric-cos-sim").textContent = `${(result.cosine_similarity * 100).toFixed(1)}%`;
            document.getElementById("metric-overall-score").textContent = `${result.overall_score}%`;

            const scoreBar = document.getElementById("coach-score-fill");
            if (scoreBar) {
                scoreBar.style.width = `${result.overall_score}%`;
                scoreBar.style.backgroundColor = result.passed ? "#1d4ed8" : "#b91c1c";
            }

            // Update Gamified XP & Streak if passed
            if (result.passed) {
                this.xpScore += 5;
                document.getElementById("gamified-xp").textContent = `${this.xpScore} XP · ${this.streakCount} Streak`;
            }

            // Render Actionable Biomechanical Correction Hints
            const hintsList = document.getElementById("coach-hints-list");
            hintsList.innerHTML = "";
            (result.actionable_hints || []).forEach((hint) => {
                const item = document.createElement("div");
                item.className = result.passed ? "coach-hint-card pass" : "coach-hint-card warn";
                item.textContent = hint;
                hintsList.appendChild(item);
            });

            // Render Joint Status Badges
            const jointGrid = document.getElementById("joint-status-grid");
            jointGrid.innerHTML = "";
            for (const [jName, status] of Object.entries(result.joint_status || {})) {
                const badge = document.createElement("span");
                badge.className = `joint-badge ${status}`;
                badge.textContent = `${jName.replace(/_/g, " ")}: ${status === "green" ? "Aligned" : "Adjust"}`;
                jointGrid.appendChild(badge);
            }
        } catch (err) {
            console.error("STEM Coach evaluation error:", err);
        }
    }

    _updateVisionTelemetryUI(metrics) {
        const elFps = document.getElementById("hud-fps");
        const elLm = document.getElementById("hud-landmarks");
        const elBuf = document.getElementById("hud-buffer");
        if (elFps) elFps.textContent = `${metrics.fps} FPS`;
        if (elLm) elLm.textContent = `${metrics.activeLandmarks} / 543 nodes`;
        if (elBuf) elBuf.textContent = `${metrics.bufferFill} / 30 frames`;
    }
}

window.addEventListener("DOMContentLoaded", () => {
    window.dhwaniDrishtiApp = new DhwaniDrishtiApp();
});
