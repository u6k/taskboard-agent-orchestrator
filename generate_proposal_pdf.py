from pathlib import Path

SRC = Path('proposal_slides.md')
OUT = Path('proposal_slides.pdf')


def escape_pdf_text(text: str) -> str:
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def parse_slides(md: str):
    chunks = [c.strip() for c in md.split('\n---\n') if c.strip()]
    slides = []
    for chunk in chunks:
        lines = [ln.rstrip() for ln in chunk.splitlines()]
        cleaned = []
        for ln in lines:
            s = ln.strip()
            if not s:
                cleaned.append('')
                continue
            if s.startswith('# '):
                cleaned.append(s[2:])
            elif s.startswith('## '):
                cleaned.append(s[3:])
            elif s.startswith('### '):
                cleaned.append(s[4:])
            elif s.startswith('- '):
                cleaned.append('• ' + s[2:])
            elif s.startswith('> '):
                cleaned.append('  ' + s[2:])
            else:
                cleaned.append(s)
        slides.append(cleaned)
    return slides


def build_content_stream(slide_lines):
    # A4 landscape: 842 x 595 points
    top = 545
    left = 46
    title_size = 24
    body_size = 15
    line_gap = 22

    cmds = ['BT']
    y = top
    for i, line in enumerate(slide_lines):
        if not line:
            y -= line_gap // 2
            continue
        size = title_size if i == 0 else body_size
        cmds.append(f'/F1 {size} Tf')
        cmds.append(f'1 0 0 1 {left} {y} Tm')
        cmds.append(f'({escape_pdf_text(line)}) Tj')
        y -= line_gap
        if y < 40:
            break
    cmds.append('ET')
    stream = '\n'.join(cmds).encode('utf-8')
    return stream


def make_pdf(slides):
    objects = []

    def add_obj(data: bytes):
        objects.append(data)
        return len(objects)

    font_id = add_obj(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')

    page_ids = []
    content_ids = []

    # Reserve pages root object later
    pages_root_id = None

    for slide in slides:
        stream = build_content_stream(slide)
        content = b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream'
        cid = add_obj(content)
        content_ids.append(cid)

        # temporary page dict; Parent set later by placeholder
        page = f'<< /Type /Page /Parent PAGES_ID 0 R /MediaBox [0 0 842 595] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {cid} 0 R >>'.encode('utf-8')
        pid = add_obj(page)
        page_ids.append(pid)

    kids = ' '.join(f'{pid} 0 R' for pid in page_ids)
    pages = f'<< /Type /Pages /Kids [ {kids} ] /Count {len(page_ids)} >>'.encode('utf-8')
    pages_root_id = add_obj(pages)

    # Replace placeholder Parent refs
    for idx, pid in enumerate(page_ids):
        raw = objects[pid - 1]
        objects[pid - 1] = raw.replace(b'PAGES_ID', str(pages_root_id).encode())

    catalog_id = add_obj(f'<< /Type /Catalog /Pages {pages_root_id} 0 R >>'.encode('utf-8'))

    out = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f'{i} 0 obj\n'.encode('utf-8'))
        out.extend(obj)
        out.extend(b'\nendobj\n')

    xref_pos = len(out)
    out.extend(f'xref\n0 {len(objects)+1}\n'.encode('utf-8'))
    out.extend(b'0000000000 65535 f \n')
    for off in offsets[1:]:
        out.extend(f'{off:010d} 00000 n \n'.encode('utf-8'))

    out.extend(b'trailer\n')
    out.extend(f'<< /Size {len(objects)+1} /Root {catalog_id} 0 R >>\n'.encode('utf-8'))
    out.extend(b'startxref\n')
    out.extend(f'{xref_pos}\n'.encode('utf-8'))
    out.extend(b'%%EOF\n')

    OUT.write_bytes(out)


if __name__ == '__main__':
    md = SRC.read_text(encoding='utf-8')
    slides = parse_slides(md)
    make_pdf(slides)
    print(f'Generated {OUT} with {len(slides)} slides.')
