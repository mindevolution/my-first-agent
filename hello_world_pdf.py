from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Create a new PDF file
pdf_file = "hello_world.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter

# Add text to the PDF
x = 100
y = height - 100
c.drawString(x, y, "Hello, World!")

# Save the PDF
c.save()

print(f"PDF created successfully: {pdf_file}")
