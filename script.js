const canvas = document.getElementById("canvas");
let currentField = null;

document.getElementById("tplInput").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;

 
  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = () => {
    canvas.innerHTML = "";
    canvas.style.width = img.naturalWidth + "px";
    canvas.style.height = img.naturalHeight + "px";
    canvas.dataset.w = img.naturalWidth;
    canvas.dataset.h = img.naturalHeight;
  };

  const fd = new FormData();
  fd.append("template", file);
  fetch("/upload-template", {
    method: "POST",
    body: fd
  })
  .then(r => r.text())
  .then(msg => {
    showToast(msg);
  
    canvas.style.backgroundImage = `url("/static/uploads/template.jpg?ts=${Date.now()}")`;
  });
});

document.getElementById("save").addEventListener("click", async () => {
  const msg = await saveLayout();
  showToast(msg);
});


document.getElementById("maskInput").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("mask", file);
  fetch("/upload-mask", { method: "POST", body: fd })
    .then(r => r.text()).then(alert);
});


function cmykToHex(c, m, y, k) {
  c /= 100; m /= 100; y /= 100; k /= 100;
  const r = 255 * (1 - c) * (1 - k),
        g = 255 * (1 - m) * (1 - k),
        b = 255 * (1 - y) * (1 - k);
  return "#" + [r, g, b].map(x => Math.round(x).toString(16).padStart(2, "0")).join("");
}

function hexToCmyk(hex) {
  const r = parseInt(hex.substr(1,2),16)/255;
  const g = parseInt(hex.substr(3,2),16)/255;
  const b = parseInt(hex.substr(5,2),16)/255;
  const k = 1 - Math.max(r, g, b);
  const c = k < 1 ? (1 - r - k) / (1 - k) : 0;
  const m = k < 1 ? (1 - g - k) / (1 - k) : 0;
  const y = k < 1 ? (1 - b - k) / (1 - k) : 0;
  return [c, m, y, k].map(v => Math.round(v * 100));
}


document.getElementById("fontColor").addEventListener("input", function () {
  const hex = this.value;
  const [c, m, y, k] = hexToCmyk(hex);
  document.getElementById("c").value = c;
  document.getElementById("m").value = m;
  document.getElementById("y").value = y;
  document.getElementById("k").value = k;
  document.getElementById("cmykColorPreview").style.backgroundColor = hex;
  applyToCurrentField({ color: hex });
  if (size) {
  currentField.dataset.size = size;
  currentField.style.fontSize = (size * 4) + "px";
}
});


["c", "m", "y", "k"].forEach(id => {
  document.getElementById(id).addEventListener("input", () => {
    const c = +document.getElementById("c").value;
    const m = +document.getElementById("m").value;
    const y = +document.getElementById("y").value;
    const k = +document.getElementById("k").value;
    const hex = cmykToHex(c, m, y, k);
    document.getElementById("fontColor").value = hex;
    document.getElementById("cmykColorPreview").style.backgroundColor = hex;
    applyToCurrentField({ color: hex });
  });
});

function addField(label = "Name", x = 50, y = 50, w = 150, h = 40) {
  const size = +document.getElementById("fontSize").value;
  const hex = document.getElementById("fontColor").value;
  const font = document.getElementById("fontFamily").value;
  const radius = +document.getElementById("borderRadius").value;
  const thickness = +document.getElementById("borderThickness").value;
  const color1 = document.getElementById("borderColor1").value;
  const color2 = document.getElementById("borderColor2").value;

  const d = document.createElement("div");
  

  const isPhoto = label.trim().toLowerCase() === "photo";
const isSignature = label.trim().toLowerCase() === "signature";
const isBarcode = label.trim().toLowerCase() === "barcode";
const isQR = label.trim().toLowerCase() === "qrcode";

d.className = "field";

if (isPhoto) {
  d.classList.add("photo");
  d.style.boxShadow = `inset 0 0 0 ${thickness / 2}px ${color1}, 0 0 0 ${thickness / 2}px ${color1}`;
  d.style.margin = `-${thickness / 2}px 0 0 -${thickness / 2}px`;
  d.style.borderRadius = radius + "px";
}

if (isSignature) {
  const img = document.createElement("img");
  img.src = "signature.png"; 
  img.style.width = "100%";
  img.style.height = "100%";
  img.style.objectFit = "contain";
  d.appendChild(img);
}

if (isBarcode) {
  const svg = document.createElement("svg");
  svg.id = `barcode-${Date.now()}`;
  d.appendChild(svg);
  setTimeout(() => {
    JsBarcode(svg, "ID123456", {
      format: "CODE128",
      width: 2,
      height: 40,
      displayValue: false
    });
  }, 0);
}

if (isQR) {
  const qrDiv = document.createElement("div");
  qrDiv.id = `qrcode-${Date.now()}`;
  d.appendChild(qrDiv);
  setTimeout(() => {
    QRCode.toCanvas(qrDiv, "https://example.com", {
      width: 80,
      margin: 0
    });
  }, 0);
}

  d.contentEditable = "true";
  d.innerText = label;
  d.style.position = "absolute";
  d.style.left = `${x}px`;
  d.style.top = `${y}px`;
  d.style.width = `${w}px`;
  d.style.height = `${h}px`;
  d.style.fontSize = `${size * 4}px`;
  // d.style.fontSize = `${size}px`;
  d.style.color = hex;
  d.style.fontFamily = getCSSFontName(font);
  d.style.textAlign = "left";

  const attrs = { x, y, w, h, size, color: hex, font, align: "left", radius, thickness, color1, color2 };
  Object.entries(attrs).forEach(([k, v]) => d.dataset[k] = v);

  d.addEventListener("click", fieldFocused);
  canvas.appendChild(d);
  currentField = d;
  fieldFocused.call(d);
  interactField(d);
}

document.querySelectorAll(".field").forEach(field => {
  field.addEventListener("click", () => {
    document.querySelectorAll(".field").forEach(f => f.classList.remove("selected"));
    currentField = field;
    field.classList.add("selected");
    updateFontSizeInput();
  });
});

document.getElementById("fontSize").addEventListener("input", function () {
  const size = parseInt(this.value);
  if (!currentField || isNaN(size)) return;

  const isPhoto = currentField.innerText.trim().toLowerCase() === "photo";
  if (isPhoto) return;

  currentField.dataset.size = size;
  currentField.style.fontSize = (size * 4) + "px";
});

function updateFontSizeInput() {
  if (!currentField) return;
  const isPhoto = currentField.innerText.trim().toLowerCase() === "photo";
  if (isPhoto) return;

  const savedSize = currentField.dataset.size || "6";
  document.getElementById("fontSize").value = savedSize;
}

function fieldFocused() {
  document.querySelectorAll(".field").forEach(f => f.classList.remove("selected"));
  this.classList.add("selected");
  currentField = this;

  const isPhoto = currentField.innerText.trim().toLowerCase() === "photo";

  if (currentField.dataset.size)
    document.getElementById("fontSize").value = currentField.dataset.size;

  if (currentField.dataset.color)
    document.getElementById("fontColor").value = currentField.dataset.color;

  if (currentField.dataset.font)
    document.getElementById("fontFamily").value = currentField.dataset.font;

  if (currentField.dataset.align)
    document.getElementById("textAlign").value = currentField.dataset.align;

  if (isPhoto) {
  document.getElementById("borderRadius").value = currentField.dataset.radius || 0;
  document.getElementById("borderThickness").value = currentField.dataset.thickness || 6;
  document.getElementById("borderColor1").value = currentField.dataset.color1 || "#ff0000";
  document.getElementById("borderColor2").value = currentField.dataset.color2 || "#0000ff";
}

  document.getElementById("fontSize").oninput = e => applyToCurrentField({ size: e.target.value });
  document.getElementById("textAlign").onchange = e => applyToCurrentField({ align: e.target.value });
  document.getElementById("fontColor").oninput = e => applyToCurrentField({ color: e.target.value });
  document.getElementById("fontFamily").onchange = e => {
  const selectedOption = e.target.selectedOptions[0];
  const fontName = selectedOption.value;
  const fontFile = selectedOption.dataset.file;

  if (currentField) {
    currentField.style.fontFamily = `'${fontName}'`;
    currentField.dataset.font = fontFile;        // 🔴 for backend (arial.ttf)
    currentField.dataset.fontname = fontName;    // 🔵 for frontend display
  }
};

  if (isPhoto) {
    document.getElementById("borderRadius").oninput = e => applyToCurrentField({ radius: e.target.value });
    document.getElementById("borderThickness").oninput = e => applyToCurrentField({ thickness: e.target.value });
    document.getElementById("borderColor1").oninput = e => applyToCurrentField({ color1: e.target.value });
    document.getElementById("borderColor2").oninput = e => applyToCurrentField({ color2: e.target.value });
  }
}

function applyToCurrentField({ color, size, font, align, radius, thickness, color1, color2 }) {
  if (!currentField) return;
  const isPhoto = currentField.innerText.trim().toLowerCase() === "photo";

  if (color) { currentField.style.color = color; currentField.dataset.color = color; }
  if (size) {
  currentField.dataset.size = size;
  currentField.style.fontSize = (size * 4) + "px";
}
  
  if (font) {
  currentField.dataset.font = font;
  currentField.style.fontFamily = `'${font}'`;
}

  if (align) { currentField.style.textAlign = align; currentField.dataset.align = align; }

  if (isPhoto) {
  if (radius !== undefined) {
    currentField.style.borderRadius = radius + "px";
    currentField.dataset.radius = radius;
  }
  if (thickness !== undefined) {
    currentField.dataset.thickness = thickness;
  }
  if (color1) currentField.dataset.color1 = color1;
  if (color2) currentField.dataset.color2 = color2;

  const t = parseFloat(currentField.dataset.thickness || 6);
  const c = currentField.dataset.color1 || "#ff0000";

  currentField.style.boxShadow = `0 0 0 ${t}px ${c}`;

   currentField.style.paddingTop = `${t / 2}px`;
  currentField.style.paddingLeft = `${t / 2}px`;
}

}

function getCSSFontName(f) {
  if (f.includes("times")) return "Times New Roman";
  if (f.includes("cour")) return "Courier New";
  return "Arial";
}

function interactField(el) {
  el.style.cursor = 'move';
  el.dataset.x = el.offsetLeft;
  el.dataset.y = el.offsetTop;

  interact(el)
    .draggable({
      inertia: true,
      modifiers: [interact.modifiers.restrictRect({ restriction: 'parent', endOnly: true })],
      listeners: {
        move(event) {
          const t = event.target;
          const x = (parseFloat(t.dataset.x) || 0) + event.dx;
          const y = (parseFloat(t.dataset.y) || 0) + event.dy;
          t.style.left = x + "px";
          t.style.top = y + "px";
          t.dataset.x = x;
          t.dataset.y = y;
        }
      }
    })
    .resizable({
      edges: { left: true, right: true, top: true, bottom: true },
      listeners: {
        move(event) {
          const t = event.target;
          t.style.width = event.rect.width + "px";
          t.style.height = event.rect.height + "px";
          t.dataset.w = event.rect.width;
          t.dataset.h = event.rect.height;
        }
      }
    });
}
async function loadFonts() {
  const fonts = [
    { file: "arial.ttf", name: "Arial" },
    { file: "arialbd.ttf", name: "Arial Bold" },
    { file: "ariali.ttf", name: "Arial Italic" },
    { file: "ariblk.ttf", name: "Arial Black" },
    { file: "calibri.ttf", name: "Calibri" },
    { file: "calibrib.ttf", name: "Calibri Bold" },
    { file: "calibrii.ttf", name: "Calibri Italic" },
    { file: "calibril.ttf", name: "Calibri Light" },
    { file: "calibrili.ttf", name: "Calibri Light Italic" },
    { file: "calibriz.ttf", name: "Calibri Bold Italic" },
    { file: "times.ttf", name: "Times New Roman" },
    { file: "timesbd.ttf", name: "Times Bold" },
    { file: "timesbi.ttf", name: "Times Bold Italic" },
    { file: "timesi.ttf", name: "Times Italic" }
  ];

  const fontSelect = document.getElementById("fontFamily");
  fonts.forEach(f => {

    const opt = document.createElement("option");
    opt.value = f.name;
    opt.textContent = f.name;
    opt.dataset.file = f.file;
    fontSelect.appendChild(opt);

    const style = document.createElement("style");
    style.innerHTML = `
      @font-face {
        font-family: '${f.name}';
        src: url('/static/fonts/${f.file}');
      }
    `;

    document.head.appendChild(style);
  });
}

document.getElementById("fontFamily").onchange = e => {
  const fontName = e.target.value;
  if (currentField) {
    currentField.style.fontFamily = `'${fontName}'`;
    currentField.dataset.font = fontName;
  }
};

loadFonts();

document.getElementById("boldBtn").addEventListener("click", () => {
  if (!currentField) return;
  currentField.style.fontWeight = currentField.style.fontWeight === "bold" ? "normal" : "bold";
});
document.getElementById("italicBtn").addEventListener("click", () => {
  if (!currentField) return;
  currentField.style.fontStyle = currentField.style.fontStyle === "italic" ? "normal" : "italic";
});

function getBorderOptions() {
  return {
    color1: document.getElementById("borderColor1").value,
    color2: document.getElementById("borderColor2").value,
    thickness: parseInt(document.getElementById("borderThickness").value),
    radius: parseInt(document.getElementById("borderRadius").value)
  };
}

function addField(label = "Name", x = 10, y = 10, w = 150, h = 30) {
  const size = +document.getElementById("fontSize").value;
  const hex = document.getElementById("fontColor").value;
  const font = document.getElementById("fontFamily").value;

  const d = document.createElement("div");
  d.className = "field";
  d.contentEditable = "true";
  d.innerText = label;
  d.style.position = "absolute";
  d.style.top = `${y}px`;
  d.style.left = `${x}px`;
  d.style.width = `${w}px`;
  d.style.height = `${h}px`;
  d.style.fontSize = `${size * 4}px`;
  d.style.color = hex;
  d.style.fontFamily = getCSSFontName(font);
  d.style.textAlign = "left";

  const attrs = { x, y, w, h, size, color: hex, font, align: "left" };
  Object.entries(attrs).forEach(([k, v]) => d.dataset[k] = v);

  d.addEventListener("focus", fieldFocused);
  canvas.appendChild(d);
  currentField = d;
  fieldFocused.call(d);
  interactField(d);
}
const loader = document.getElementById("loader");

// Save layout
async function saveLayout() {
  const fields = Array.from(document.querySelectorAll(".field")).map(f => {
    const label = f.innerHTML.trim();
    const field = {
      label: label,
      x: parseInt(f.style.left),
      y: parseInt(f.style.top),
      w: parseInt(f.style.width),
      h: parseInt(f.style.height),
      size: +f.dataset.size || 24,
      color: f.dataset.color || "#000000",
      font: f.dataset.font || "arial.ttf",
      fontname: f.dataset.fontname || "Arial",
      align: f.dataset.align || "left"
    };

    if (label.toLowerCase() === "photo") {
      field.radius = +f.dataset.radius || 0;
      field.thickness = +f.dataset.thickness || 6;
      field.color1 = f.dataset.color1 || "#ff0000";
      field.color2 = f.dataset.color2 || "#0000ff";
    }

    return field;
  });

  const layout = {
    template_size: {
      w: +canvas.dataset.w,
      h: +canvas.dataset.h
    },
    fields
  };

  console.log("Layout before saving:", layout); // ✅ yeh line layout check karne ke liye hai

  const res = await fetch("/save-layout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(layout)
  });

  const msg = await res.text();
  return msg;
}

document.getElementById("previewBtn").addEventListener("click", async () => {
  const loader = document.getElementById("loader");
  const container = document.getElementById("pdfPreviewContainer");
  const iframe = document.getElementById("pdfPreview");

  await saveLayout();

  loader.style.display = "inline-block";

  try {
    const ts = Date.now();
    const res = await fetch(`/generate-sample?ts=${ts}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    iframe.src = url;
    container.style.display = "block";
  } catch (err) {
    alert("❌ Preview load failed");
  } finally {
    loader.style.display = "none";
  }
});

  document.getElementById("closePreview").onclick = () => {
  document.getElementById("pdfPreviewContainer").style.display = "none";
};

document.getElementById("downloadCards").onclick = () => {
  downloadZip("/download-cards", "Cards");
};

document.getElementById("downloadSheets").onclick = () => {
  downloadZip("/download-sheets", "Sheets");
};

async function downloadZip(url, name) {
  loader.style.display = "inline-block";

  try {
      await fetch("/generate-all");

    const res = await fetch(url);
    if (!res.ok) throw new Error("ZIP download failed");

    const blob = await res.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${name}.zip`;
    link.click();

    showToast(`✅ ${name} ZIP downloaded`);
  } catch (e) {
    showToast(`❌ ${name} download failed`);
  } finally {
    loader.style.display = "none";
  }
}

function showToast(message) {
  toast.innerText = message;
  toast.style.display = "block";
  toast.style.opacity = "1";
  toast.style.background = message.includes("✅") ? "#4caf50" : "#f44336";
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => {
      toast.style.display = "none";
    }, 500);
  }, 3000);
}


function showLoader() {
  document.getElementById("loader").style.display = "inline-block";
  loader.style.display = "inline-block";
  loader.querySelector('.loader-outer').classList.add("spin-outer");
  loader.querySelector('.loader-inner').classList.add("spin-inner");
}

function hideLoader() {
  document.getElementById("loader").style.display = "none";
  loader.style.display = "none";
  loader.querySelector('.loader-outer').classList.remove("spin-outer");
  loader.querySelector('.loader-inner').classList.remove("spin-inner");
}


document.getElementById("downloadBtn").onclick = () => {
  window.location.href = "/generate-sample?ts=" + Date.now();
};
