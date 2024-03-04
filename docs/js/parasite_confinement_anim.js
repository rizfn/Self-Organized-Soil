const numDigits = 1024;
const totalChunks = 10000
const canvas = document.querySelector('#c');
canvas.width = numDigits;
canvas.height = numDigits;
canvas.style.imageRendering = 'pixelated';

const smallerDimension = Math.min(window.innerWidth, window.innerHeight);
canvas.style.width = `${smallerDimension}px`;
canvas.style.height = `${smallerDimension}px`;

const ctx = canvas.getContext('2d');

const portionSize = numDigits * numDigits;
let portion = new Uint8Array(portionSize);
let currentChunk = 0;

let isPlaying = true;

document.getElementById('playPauseButton').addEventListener('click', function() {
    isPlaying = !isPlaying;
    this.textContent = isPlaying ? 'Pause' : 'Play';
});


function fetchAndDisplayNextChunk() {
    if (!isPlaying) {
        return;
    }
    fetch('../data/twospec_samenutrient/lattice_anim_L_1024_sigma_1_theta_0.042_rhofactor_4.bin', {
        headers: {
            'Range': `bytes=${currentChunk * portionSize}-${(currentChunk + 1) * portionSize - 1}`
        }
    })
    .then(response => response.arrayBuffer())
    .then(data => {
        portion = new Uint8Array(data);
        displayTimestep(portion);
        currentChunk++;
        if (currentChunk >= totalChunks) {
            currentChunk = 0; // Reset to the first chunk when we reach the end
        }
    });
}

// Fetch and display the first chunk immediately
fetchAndDisplayNextChunk();

// Then fetch and display a new chunk every 10ms
setInterval(fetchAndDisplayNextChunk, 10);

function displayTimestep(portion) {
   let imageData = ctx.createImageData(numDigits, numDigits);
    for (let row = 0; row < numDigits; row++) {
        for (let col = 0; col < numDigits; col++) {
            const pixelValue = portion[row * numDigits + col];
            let index = (row * numDigits + col) * 4;
            switch (pixelValue) {
                case 0:
                    imageData.data.set([224, 224, 224, 255], index);
                    break;
                case 1:
                    imageData.data.set([150, 200, 200, 255], index);
                    break;
                case 2:
                    imageData.data.set([102, 51, 0, 200], index);
                    break;
                case 3:
                    imageData.data.set([0, 150, 0, 255], index);
                    break;
                case 4:
                    imageData.data.set([50, 50, 150, 255], index);
                    break;
                default:
                    imageData.data.set([255, 0, 0, 255], index);
                    break;
            }
        }
    }
    ctx.putImageData(imageData, 0, 0);
}