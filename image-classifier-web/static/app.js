const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const resultDiv = document.getElementById('result');

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
});
