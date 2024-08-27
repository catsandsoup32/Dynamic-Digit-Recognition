
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

let isDrawing = false;
let lastX = 0;
let lastY = 0;


// Start drawing when mouse button is pressed
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    lastX = e.offsetX;
    lastY = e.offsetY;
});
// Draw on canvas while mouse is moving
canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    
    lastX = e.offsetX;
    lastY = e.offsetY;
});
// Stop drawing when mouse button is released
canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

// -------------------------------------------
// Event listeners
document.getElementById('clear').addEventListener('click', clearScreen);


// -------------------------------------------
// Functions

function clearScreen(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}




