function showImage() {
    const elements = document.getElementsByClassName('embedding-plot');
    for (let i = 0; i < elements.length; i++) {
        elements[i].style.display = 'block'; // Make each element visible
        console.log(elements[i]);
    }
}
