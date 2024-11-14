document.addEventListener('DOMContentLoaded', () => {
    // Multi-Step Form Navigation
    const steps = Array.from(document.querySelectorAll('.form-step'));  // All form steps
    const nextBtns = document.querySelectorAll('.next-btn');            // All "Next" buttons
    const backBtns = document.querySelectorAll('.back-btn');            // All "Back" buttons
    let currentStep = 0;  // Track the current step

    function showStep(step) {
        steps.forEach((stepDiv, index) => {
            // Only show the active step and add transition effect
            stepDiv.classList.toggle('active', index === step);
            stepDiv.style.transition = 'opacity 0.3s ease-in-out';
        });

        // Disable "Back" button on the first step
        backBtns.forEach(btn => btn.style.display = step === 0 ? 'none' : 'inline-block');

        // Hide "Next" button on the last step
        nextBtns.forEach(btn => btn.style.display = step === steps.length - 1 ? 'none' : 'inline-block');
    }

    // Event listeners for "Next" buttons
    nextBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            if (currentStep < steps.length - 1) {
                currentStep += 1;
                showStep(currentStep);  // Move to the next step
            }
        });
    });

    // Event listeners for "Back" buttons
    backBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            if (currentStep > 0) {
                currentStep -= 1;
                showStep(currentStep);  // Move to the previous step
            }
        });
    });

    showStep(currentStep);  // Show the first step initially
});
