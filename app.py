import gradio as gr
import torch, math
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_NAME = "aailarehman/factsense-fake-news-detector"
tokenizer  = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model      = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

def make_gauge_svg(real_pct, fake_pct, is_real):
    needle_ratio = real_pct / 100
    nx = 150 + 90 * math.cos(math.pi + needle_ratio * math.pi)
    ny = 130 + 90 * math.sin(math.pi + needle_ratio * math.pi)
    gauge_color = "#16a34a" if is_real else "#dc2626"
    label       = "REAL NEWS" if is_real else "FAKE NEWS"
    main_pct    = real_pct if is_real else fake_pct
    return f"""<svg viewBox="0 0 300 155" xmlns="http://www.w3.org/2000/svg"
         style="width:100%;max-width:260px;display:block;margin:0 auto">
      <path d="M 20 130 A 130 130 0 0 1 80 27"  fill="#fca5a5" stroke="none"/>
      <path d="M 80 27  A 130 130 0 0 1 220 27" fill="#fde68a" stroke="none"/>
      <path d="M 220 27 A 130 130 0 0 1 280 130" fill="#86efac" stroke="none"/>
      <circle cx="150" cy="130" r="70" fill="white"/>
      <line x1="150" y1="130" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="#0f172a" stroke-width="3" stroke-linecap="round"/>
      <circle cx="150" cy="130" r="6" fill="#0f172a"/>
      <text x="150" y="113" text-anchor="middle"
            font-family="Inter,system-ui" font-size="20" font-weight="800"
            fill="{gauge_color}">{main_pct:.1f}%</text>
      <text x="150" y="128" text-anchor="middle"
            font-family="Inter,system-ui" font-size="9" font-weight="600"
            fill="#94a3b8">{label}</text>
      <text x="22"  y="150" font-family="Inter,system-ui" font-size="9" fill="#94a3b8">Fake</text>
      <text x="122" y="152" font-family="Inter,system-ui" font-size="9" fill="#94a3b8">Uncertain</text>
      <text x="258" y="150" font-family="Inter,system-ui" font-size="9" fill="#94a3b8">Real</text>
    </svg>"""

def build_result_html(fake_pct, real_pct):
    if fake_pct is None:
        return """<div style="background:#f8fafc;border:1px solid #e2e8f0;
                    border-radius:14px;padding:40px 24px;text-align:center;
                    font-family:Inter,system-ui;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;height:100%">
          <div style="font-size:14px;font-weight:600;color:#64748b">Awaiting analysis</div>
          <div style="font-size:12px;color:#94a3b8;margin-top:5px">
              Paste an article and click Analyze</div></div>"""
    is_real   = real_pct > fake_pct
    label     = "Real News" if is_real else "Fake News"
    main_pct  = real_pct if is_real else fake_pct
    txt_color = "#15803d" if is_real else "#b91c1c"
    dot_color = "#16a34a" if is_real else "#dc2626"
    gauge_svg = make_gauge_svg(real_pct, fake_pct, is_real)
    real_w    = f"{real_pct:.1f}%"
    fake_w    = f"{fake_pct:.1f}%"
    return f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
                padding:20px;font-family:Inter,system-ui,sans-serif;
                display:flex;flex-direction:column;gap:14px;height:100%">
      <div style="font-size:10px;font-weight:700;color:#94a3b8;
                  text-transform:uppercase;letter-spacing:0.08em">Confidence meter</div>
      {gauge_svg}
      <div>
        <div style="display:flex;justify-content:space-between;
                    font-size:12px;font-weight:700;margin-bottom:5px">
          <span style="color:#475569">Real probability</span>
          <span style="color:#15803d">{real_pct:.1f}%</span>
        </div>
        <div style="height:8px;background:#f1f5f9;border-radius:4px;overflow:hidden">
          <div style="height:100%;width:{real_w};background:#16a34a;border-radius:4px"></div>
        </div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;
                    font-size:12px;font-weight:700;margin-bottom:5px">
          <span style="color:#475569">Fake probability</span>
          <span style="color:#dc2626">{fake_pct:.1f}%</span>
        </div>
        <div style="height:8px;background:#f1f5f9;border-radius:4px;overflow:hidden">
          <div style="height:100%;width:{fake_w};background:#dc2626;border-radius:4px"></div>
        </div>
      </div>
      <div style="border-top:1px solid #f1f5f9;padding-top:14px;
                  display:flex;align-items:center;gap:10px">
        <div style="width:10px;height:10px;border-radius:50%;
                    background:{dot_color};flex-shrink:0"></div>
        <div style="font-size:18px;font-weight:800;color:{txt_color}">{label}</div>
        <div style="font-size:18px;font-weight:800;color:{txt_color};
                    margin-left:auto">{main_pct:.1f}%</div>
      </div>
    </div>"""

def predict_news(text):
    if not text or len(text.strip()) < 20:
        return build_result_html(None, None)
    inputs = tokenizer(text, max_length=256, padding="max_length",
                       truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
    fake_pct = round(float(probs[0]) * 100, 1)
    real_pct = round(float(probs[1]) * 100, 1)
    return build_result_html(fake_pct, real_pct)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { box-sizing: border-box; }
.gradio-container {
    max-width: 1020px !important; margin: 0 auto !important;
    font-family: Inter, system-ui, sans-serif !important;
    background: #f1f5f9 !important; padding: 16px !important;
}
footer { display: none !important; }
#topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 12px 18px; margin-bottom: 14px;
}
.brand-block { display: flex; align-items: center; gap: 10px; }
.brand-icon { width: 34px; height: 34px; background: #0f172a; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 15px; }
.brand-name { font-size: 16px; font-weight: 800; color: #0f172a; }
.brand-sub  { font-size: 11px; color: #94a3b8; margin-top: 1px; }
.bdg-row { display: flex; gap: 6px; }
.bdg { font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 6px; }
.bdg-dark  { background: #0f172a; color: #f8fafc; }
.bdg-green { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
.bdg-slate { background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; }
.gr-group { background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important; padding: 16px !important; }
textarea { background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important; font-size: 13px !important;
    color: #334155 !important; line-height: 1.6 !important;
    padding: 12px !important; resize: none !important; }
textarea:focus { border-color: #94a3b8 !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(148,163,184,0.2) !important; }
#analyze-btn button { background: #0f172a !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 700 !important;
    padding: 10px !important; width: 100% !important; }
#analyze-btn button:hover { background: #1e293b !important; }
#clear-btn button { background: #fff !important; color: #374151 !important;
    border: 1px solid #e2e8f0 !important; border-radius: 8px !important;
    font-size: 13px !important; padding: 10px !important; width: 100% !important; }
.ex-grid { display: grid; grid-template-columns: repeat(4,1fr);
    gap: 8px; margin-top: 10px; }
.ex-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 10px 12px; cursor: pointer; transition: border-color 0.15s; }
.ex-card:hover { border-color: #94a3b8; background: #fff; }
.ex-type { font-size: 9px; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 5px;
    display: flex; align-items: center; gap: 4px; }
.ext-fake { color: #b91c1c; } .ext-real { color: #15803d; }
.ex-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.exd-fake { background: #dc2626; } .exd-real { background: #16a34a; }
.ex-text { font-size: 11px; color: #64748b; line-height: 1.4; }
"""

with gr.Blocks(css=css, title="FactSense — Fake News Detector") as demo:
    gr.HTML("""<div id="topbar">
      <div class="brand-block">
        <div class="brand-icon">🔍</div>
        <div><div class="brand-name">FactSense</div>
        <div class="brand-sub">AI Misinformation Detection Platform</div></div>
      </div>
      <div class="bdg-row">
        <span class="bdg bdg-dark">DistilBERT</span>
        <span class="bdg bdg-green">98.8% accuracy</span>
        <span class="bdg bdg-slate">116k articles</span>
      </div></div>""")
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            with gr.Group():
                gr.HTML("<p style=\"font-size:10px;font-weight:700;color:#94a3b8;"
                        "text-transform:uppercase;letter-spacing:0.08em;margin:0 0 10px 0\">"
                        "Article input</p>")
                text_input = gr.Textbox(
                    placeholder="Paste any news article, headline, or claim here...",
                    lines=9, show_label=False, show_copy_button=True)
                with gr.Row():
                    clear_btn  = gr.Button("Clear",             elem_id="clear-btn",  scale=1)
                    submit_btn = gr.Button("Analyze Article →", elem_id="analyze-btn",scale=3)
        with gr.Column(scale=2):
            result_out = gr.HTML(value=build_result_html(None, None))
    with gr.Group():
        gr.HTML("""<p style="font-size:10px;font-weight:700;color:#94a3b8;
                  text-transform:uppercase;letter-spacing:0.08em;margin:0 0 10px 0">
                  Example articles — click to load</p>
        <div class="ex-grid">
          <div class="ex-card" onclick="var t=document.querySelector(\'textarea\');t.value=\'BREAKING: Bill Gates admits microchips are being secretly inserted into COVID vaccines. Leaked documents confirm the global surveillance agenda. Share before this gets deleted!\';t.dispatchEvent(new Event(\'input\',{bubbles:true}))">
            <div class="ex-type ext-fake"><div class="ex-dot exd-fake"></div>Fake</div>
            <div class="ex-text">Bill Gates admits microchips inserted into vaccines. Leaked documents confirm surveillance...</div></div>
          <div class="ex-card" onclick="var t=document.querySelector(\'textarea\');t.value=\'URGENT: Scientists discovered 5G towers emit radiation destroying human DNA. The government is covering this up to protect telecom profits. Warn your family now!\';t.dispatchEvent(new Event(\'input\',{bubbles:true}))">
            <div class="ex-type ext-fake"><div class="ex-dot exd-fake"></div>Fake</div>
            <div class="ex-text">5G towers emit radiation destroying human DNA. Government covering up for telecom profits...</div></div>
          <div class="ex-card" onclick="var t=document.querySelector(\'textarea\');t.value=\'The European Central Bank raised its key interest rate by half a percentage point on Thursday in its ongoing effort to combat inflation across the eurozone.\';t.dispatchEvent(new Event(\'input\',{bubbles:true}))">
            <div class="ex-type ext-real"><div class="ex-dot exd-real"></div>Real</div>
            <div class="ex-text">European Central Bank raised interest rate half point to combat eurozone inflation...</div></div>
          <div class="ex-card" onclick="var t=document.querySelector(\'textarea\');t.value=\'Apple Inc reported record quarterly earnings of 119.6 billion dollars in revenue surpassing Wall Street expectations. CFO Luca Maestri attributed results to strong services growth.\';t.dispatchEvent(new Event(\'input\',{bubbles:true}))">
            <div class="ex-type ext-real"><div class="ex-dot exd-real"></div>Real</div>
            <div class="ex-text">Apple reported record quarterly earnings of 119.6 billion surpassing Wall Street expectations...</div></div>
        </div>""")
    gr.HTML("""<div style="display:flex;align-items:center;justify-content:space-between;
                margin-top:12px;padding-top:12px;border-top:1px solid #e2e8f0;
                font-size:11px;color:#94a3b8;font-family:Inter,system-ui">
      <span>Built with PyTorch · HuggingFace Transformers · Gradio · ISOT + WELFake</span>
      <span style="background:#f1f5f9;color:#475569;padding:3px 10px;
                   border-radius:20px;font-weight:500">distilbert-base-uncased</span>
    </div>""")
    submit_btn.click(fn=predict_news, inputs=text_input, outputs=result_out)
    clear_btn.click(fn=lambda: ("", build_result_html(None, None)),
                    outputs=[text_input, result_out])

demo.launch()
