export function initControlNetModals(container) {
    // Get all the buttons that open a modal
    const btns = container.querySelectorAll(".cnet-modal-open");

    // Get all the <span> elements that close a modal
    const spans = container.querySelectorAll(".cnet-modal-close");

    // For each button, add a click event listener that opens the corresponding modal
    btns.forEach((btn) => {
        const modalId = btn.id.replace('cnet-modal-open-', '');
        const modal = container.querySelector("#cnet-modal-" + modalId);
        btn.addEventListener('click', () => {
            modal.style.display = "block";
        });
    });

    // For each <span> element, add a click event listener that closes the corresponding modal
    spans.forEach((span) => {
        const modal = span.parentNode;
        span.addEventListener('click', () => {
            modal.style.display = "none";
        });
    });
}
