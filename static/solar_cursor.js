(function () {
    function setupSolarGoldCursor() {
        const supportsFinePointer = window.matchMedia("(pointer: fine)").matches;
        const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        if (!supportsFinePointer || prefersReducedMotion) {
            return;
        }

        document.querySelectorAll(".solar-cursor-dot, .solar-cursor-ring").forEach((el) => {
            el.remove();
        });

        const dot = document.createElement("div");
        const ring = document.createElement("div");

        dot.className = "solar-cursor-dot";
        ring.className = "solar-cursor-ring";

        document.body.appendChild(ring);
        document.body.appendChild(dot);
        document.body.classList.add("solar-cursor-enabled");

        let mouseX = window.innerWidth / 2;
        let mouseY = window.innerHeight / 2;

        let ringX = mouseX;
        let ringY = mouseY;

        let ringScale = 1;
        let targetScale = 1;

        const interactiveSelector = [
            "a",
            "button",
            "input",
            "select",
            "textarea",
            "[role='button']",
            ".nav-item",
            ".primary-btn",
            ".ghost-btn",
            ".logout-btn",
            ".profile-link-btn",
            ".kpi-card",
            ".panel",
            ".quick-card",
            ".hero-card",
            ".theme-toggle-btn",
            ".filter-btn"
        ].join(",");

        function setDotPosition(x, y) {
            dot.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
        }

        function setRingPosition(x, y, scale) {
            ring.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%) scale(${scale})`;
        }

        window.addEventListener(
            "mousemove",
            function (event) {
                mouseX = event.clientX;
                mouseY = event.clientY;

                document.body.classList.add("solar-cursor-visible");

                setDotPosition(mouseX, mouseY);
            },
            { passive: true }
        );

        window.addEventListener("mouseenter", function () {
            document.body.classList.add("solar-cursor-visible");
        });

        window.addEventListener("mouseleave", function () {
            document.body.classList.remove("solar-cursor-visible");
        });

        document.addEventListener(
            "mouseover",
            function (event) {
                if (event.target.closest(interactiveSelector)) {
                    document.body.classList.add("solar-cursor-hover");
                    targetScale = 1.5;
                }
            },
            { passive: true }
        );

        document.addEventListener(
            "mouseout",
            function (event) {
                if (event.target.closest(interactiveSelector)) {
                    document.body.classList.remove("solar-cursor-hover");
                    targetScale = 1;
                }
            },
            { passive: true }
        );

        window.addEventListener("mousedown", function () {
            document.body.classList.add("solar-cursor-active");
            targetScale = 0.62;
        });

        window.addEventListener("mouseup", function () {
            document.body.classList.remove("solar-cursor-active");

            if (document.body.classList.contains("solar-cursor-hover")) {
                targetScale = 1.5;
            } else {
                targetScale = 1;
            }
        });

        function animateRing() {
            ringX += (mouseX - ringX) * 0.22;
            ringY += (mouseY - ringY) * 0.22;
            ringScale += (targetScale - ringScale) * 0.22;

            setRingPosition(ringX, ringY, ringScale);

            requestAnimationFrame(animateRing);
        }

        animateRing();
    }

    document.addEventListener("DOMContentLoaded", setupSolarGoldCursor);
})();