from PIL import Image, ImageDraw, ImageFont
import os

# Cargar la imagen original
img = Image.open('C:/Users/usuario/Desktop/1.png').convert('RGB')
width, height = img.size
print(f"Imagen: {width}x{height}")

# Color original del fondo de la imagen
old_bg = (11, 75, 131)

# Color del sidebar del dashboard (#4a6fa5)
new_bg = (74, 111, 165)

# Reemplazar el color de fondo en toda la imagen
pixels = img.load()
tolerance = 15  # Tolerancia para detectar el color de fondo

for y in range(height):
    for x in range(width):
        r, g, b = pixels[x, y]
        # Si el pixel es similar al fondo original, cambiarlo al nuevo
        if (abs(r - old_bg[0]) < tolerance and
            abs(g - old_bg[1]) < tolerance and
            abs(b - old_bg[2]) < tolerance):
            pixels[x, y] = new_bg

print("Fondo cambiado de", old_bg, "a", new_bg)

draw = ImageDraw.Draw(img)

# Borrar "pescaderías" con el nuevo color de fondo
draw.rectangle((2300, 1280, 4300, 1800), fill=new_bg)
print("Borrado pescaderías")

# Color del texto "inversiones" - ajustado para el nuevo fondo
text_color = (200, 215, 235)

# Cargar fuente Georgia Bold Italic
font = ImageFont.truetype('C:/Windows/Fonts/georgiaz.ttf', 270)

# Dibujar "inversiones"
text = "inversiones"
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]

x_pos = 4050 - text_width
y_pos = 1370

draw.text((x_pos, y_pos), text, font=font, fill=text_color)
print(f"Texto en ({x_pos}, {y_pos})")

# Guardar como PNG
img.save('web/static/logo_carihuela.png', 'PNG')
print("Logo guardado: web/static/logo_carihuela.png")
