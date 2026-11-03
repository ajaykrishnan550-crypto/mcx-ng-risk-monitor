import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from risk_engine import analyse

st.set_page_config(page_title="MCX Natural Gas Position Risk Monitor", page_icon="⛽", layout="wide")
st.title("MCX Natural Gas · Screenshot & Position Risk Monitor")
st.caption("Upload a Groww option-chain screenshot, verify the extracted data, enter your open positions, and review your risk.")

# ---------- helpers ----------
def empty_chain():
    return pd.DataFrame([
        {"strike": 280, "ce_ltp": 0.0, "ce_oi": 0, "ce_iv": np.nan, "pe_oi": 0, "pe_ltp": 0.0, "pe_iv": np.nan},
        {"strike": 290, "ce_ltp": 0.0, "ce_oi": 0, "ce_iv": np.nan, "pe_oi": 0, "pe_ltp": 0.0, "pe_iv": np.nan},
        {"strike": 300, "ce_ltp": 0.0, "ce_oi": 0, "ce_iv": np.nan, "pe_oi": 0, "pe_ltp": 0.0, "pe_iv": np.nan},
    ])

def chain_to_long(df, expiry):
    rows=[]
    for _,r in df.iterrows():
        for side,ltp,oi,iv in [("CE",r.get("ce_ltp"),r.get("ce_oi"),r.get("ce_iv")),("PE",r.get("pe_ltp"),r.get("pe_oi"),r.get("pe_iv"))]:
            rows.append({"expiry":expiry,"strike":r.get("strike"),"option_type":side,"ltp":ltp,"oi":oi,"iv":iv})
    return pd.DataFrame(rows)

def parse_groww_screenshot(upload):
    """Parse the visible Groww layout: CE OI | CE price | Strike | PE price | PE OI."""
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR
    import io, re
    img=Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
    detections,_=RapidOCR()(np.array(img))
    if not detections:
        raise ValueError("No readable text found. Upload a clear, uncropped Groww option-chain screenshot.")
    items=[]
    for box,txt,score in detections:
        xs=[pt[0] for pt in box]; ys=[pt[1] for pt in box]
        cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2
        clean=str(txt).replace(",", "").replace("₹", "").strip()
        items.append({"x":cx,"y":cy,"text":clean,"score":float(score)})
    # Strike labels are in the centre column and have a .00 suffix.
    strikes=[]
    for it in items:
        m=re.fullmatch(r"(\d{2,3})\.00",it["text"])
        if m and 300 <= it["x"] <= 620:
            strikes.append((float(m.group(1)),it["y"]))
    if not strikes:
        raise ValueError("Could not identify strike rows. Make sure the strike column is visible.")
    rows=[]
    for strike,y in sorted(strikes):
        row={"strike":strike,"ce_ltp":np.nan,"ce_oi":np.nan,"ce_iv":np.nan,"pe_oi":np.nan,"pe_ltp":np.nan,"pe_iv":np.nan}
        candidates=[it for it in items if abs(it["y"]-y)<=24 and it["text"] not in ("—","-")]
        for it in candidates:
            t=it["text"]
            if not re.fullmatch(r"\d+(?:\.\d+)?",t): continue
            val=float(t); x=it["x"]
            if x < 135 and val.is_integer(): row["ce_oi"]=int(val)
            elif 135 <= x < 300 and val < 200: row["ce_ltp"]=val
            elif 620 <= x < 770 and val < 200: row["pe_ltp"]=val
            elif x >= 770 and val.is_integer(): row["pe_oi"]=int(val)
        rows.append(row)
    df=pd.DataFrame(rows).drop_duplicates(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    if df[["ce_oi","ce_ltp","pe_ltp","pe_oi"]].notna().sum().sum()<len(df)*2:
        raise ValueError("OCR found strikes but too few option values. Please verify manually or upload a sharper screenshot.")
    return df

with st.sidebar:
    st.header("1 · Groww screenshot")
    shot=st.file_uploader("Upload option-chain screenshot", type=["png","jpg","jpeg","webp"], key="chain_shot")
    if shot:
        st.image(shot, caption="Uploaded Groww screenshot", use_container_width=True)
        if st.button("Extract option chain from screenshot", type="primary"):
            try:
                st.session_state.chain_data=parse_groww_screenshot(shot)
                st.session_state["ocr_status"]="Extraction completed. Check every value in the editable table before analysis."
            except Exception as e:
                st.session_state["ocr_status"]=str(e)
    if st.session_state.get("ocr_status"):
        st.info(st.session_state["ocr_status"])
    st.divider()
    st.header("2 · Contract details")
    expiry=st.text_input("Expiry / contract month", value="Current month")
    fut_price=st.number_input("Current futures price", min_value=0.0, value=0.0, step=0.5)
    previous_price=st.number_input("Previous snapshot futures price (optional)", min_value=0.0, value=0.0, step=0.5)
    fut_file=st.file_uploader("Optional futures history CSV (hourly OHLCV + OI)", type=["csv"], key="fut_csv")

st.subheader("3 · Verify option-chain numbers from the screenshot")
st.write("The extractor is tuned for this Groww layout: Call OI, Call price, Strike, Put price, Put OI. OCR can misread digits, so verify the editable rows before analysing.")
if "chain_data" not in st.session_state: st.session_state.chain_data=empty_chain()
chain=st.data_editor(st.session_state.chain_data, num_rows="dynamic", use_container_width=True, key="chain_editor", column_config={
    "strike":st.column_config.NumberColumn("Strike", required=True),
    "ce_ltp":st.column_config.NumberColumn("CE LTP", min_value=0.0),
    "ce_oi":st.column_config.NumberColumn("CE OI", min_value=0),
    "ce_iv":st.column_config.NumberColumn("CE IV %"),
    "pe_oi":st.column_config.NumberColumn("PE OI", min_value=0),
    "pe_ltp":st.column_config.NumberColumn("PE LTP", min_value=0.0),
    "pe_iv":st.column_config.NumberColumn("PE IV %"),
})
st.session_state.chain_data=chain

st.subheader("4 · Your existing short positions")
st.write("Add every open short CE and PE. Quantity means the total number of units, not number of lots. Enter the latest LTP from Groww to calculate approximate mark-to-market P&L.")
if "positions" not in st.session_state:
    st.session_state.positions=pd.DataFrame([{"side":"PE","strike":np.nan,"sell_price":np.nan,"current_ltp":np.nan,"quantity":np.nan,"expiry":"Current month"},{"side":"CE","strike":np.nan,"sell_price":np.nan,"current_ltp":np.nan,"quantity":np.nan,"expiry":"Current month"}])
positions=st.data_editor(st.session_state.positions, num_rows="dynamic", use_container_width=True, key="positions_editor", column_config={
    "side":st.column_config.SelectboxColumn("Option side", options=["PE","CE"], required=True),
    "strike":st.column_config.NumberColumn("Strike", min_value=0.0),
    "sell_price":st.column_config.NumberColumn("Your sell price", min_value=0.0),
    "current_ltp":st.column_config.NumberColumn("Current LTP", min_value=0.0),
    "quantity":st.column_config.NumberColumn("Quantity (units)", min_value=1, step=1),
    "expiry":st.column_config.TextColumn("Expiry"),
})
st.session_state.positions=positions

if st.button("Analyse my positions and current-month opportunity", type="primary", use_container_width=True):
    try:
        chain=chain.copy()
        for c in chain.columns: chain[c]=pd.to_numeric(chain[c],errors="coerce")
        chain=chain.dropna(subset=["strike"])
        if chain.empty: raise ValueError("Enter at least one strike row from the screenshot.")
        if fut_price<=0: raise ValueError("Enter the current futures price.")
        opt=chain_to_long(chain,expiry)
        if fut_file:
            fut=pd.read_csv(fut_file)
            if "close" not in fut: raise ValueError("Futures CSV needs a close column.")
            result=analyse(opt,fut)
            put_risk=result["put_score"]; call_risk=result["call_score"]
            trend=result["trend"]
        else:
            result={"pcr":None,"atm_strike":np.nan,"atm_straddle":np.nan,"atm_iv":np.nan,"warnings":[]}
            put_risk=call_risk= None; trend="NOT ENOUGH HISTORY"
        # Use the entered current futures price for ATM calculations and PCR from uploaded chain.
        pe_oi=pd.to_numeric(chain.pe_oi,errors="coerce").fillna(0).sum()
        ce_oi=pd.to_numeric(chain.ce_oi,errors="coerce").fillna(0).sum()
        pcr=pe_oi/ce_oi if ce_oi else np.nan
        atm_idx=(chain.strike-fut_price).abs().idxmin(); atm=chain.loc[atm_idx]
        atm_straddle=float(atm.ce_ltp or 0)+float(atm.pe_ltp or 0)
        # Position P&L: short premium received minus current buyback value, multiplied by quantity.
        pos=positions.copy()
        for c in ["sell_price","current_ltp","quantity","strike"]: pos[c]=pd.to_numeric(pos[c],errors="coerce")
        pos["unrealised_pnl"]=(pos.sell_price-pos.current_ltp)*pos.quantity
        pos["risk_side"]=pos.side.map({"PE":"Put seller","CE":"Call seller"})
        st.session_state["last_report"]={"result":result,"positions":pos,"pcr":pcr,"atm_strike":float(atm.strike),"atm_straddle":atm_straddle,"put_risk":put_risk,"call_risk":call_risk,"trend":trend,"price":fut_price,"expiry":expiry,"previous_price":previous_price}
    except Exception as e: st.error(str(e))

report=st.session_state.get("last_report")
if report:
    st.divider(); st.header("Analysis report")
    a,b,c,d=st.columns(4)
    a.metric("Current futures",f"{report['price']:.2f}")
    b.metric("PCR (OI)","—" if pd.isna(report["pcr"]) else f"{report['pcr']:.2f}")
    c.metric("ATM strike",f"{report['atm_strike']:.2f}")
    d.metric("ATM straddle",f"{report['atm_straddle']:.2f}")
    st.subheader("Your open-position P&L")
    p=report["positions"]
    st.dataframe(p[["side","strike","sell_price","current_ltp","quantity","unrealised_pnl"]],use_container_width=True)
    total=float(p.unrealised_pnl.sum()) if len(p) else 0.0
    st.metric("Estimated combined unrealised P&L",f"₹ {total:,.2f}")
    st.caption("This is a premium-based estimate before brokerage, taxes, slippage and any contract-specific adjustments. Verify quantity and LTP.")
    st.subheader("Which side looks relatively safer to sell?")
    if report["put_risk"] is None:
        st.warning("Upload a futures-history CSV with preferably 50–100+ hourly candles to calculate trend, EMA, ATR, ADX, volume and OI-based directional risk. Without it, the app will not pretend to know whether calls or puts are safer.")
    else:
        st.write(f"Futures trend: **{report['trend']}**")
        x,y=st.columns(2)
        x.metric("Short PUT risk",f"{report['put_risk']}/100")
        y.metric("Short CALL risk",f"{report['call_risk']}/100")
        if report["put_risk"]+10 < report["call_risk"]: st.success("Current technical readings favour PUT-side selling relatively more than CALL-side selling. This is not a sell signal.")
        elif report["call_risk"]+10 < report["put_risk"]: st.success("Current technical readings favour CALL-side selling relatively more than PUT-side selling. This is not a sell signal.")
        else: st.warning("No clear technical advantage between PUT and CALL selling. Avoid forcing a trade when the risk scores are close.")
        for w in report["result"].get("warnings",[]): st.warning(w)
    st.info("The side comparison is a risk-screening result, not a forecast. It does not account for all expiry-specific Greeks, margin, gap risk or event risk. Do not average losing naked options based only on this score.")
    export={"generated_at":datetime.now().isoformat(),"expiry":report["expiry"],"market":{k:v for k,v in report.items() if k not in ["positions","result"]},"positions":report["positions"].to_dict(orient="records"),"technical_result":report["result"]}
    st.download_button("Download this analysis report (JSON)",json.dumps(export,indent=2,default=str),file_name="mcx_ng_position_report.json",mime="application/json")

st.divider()
st.caption("Important: Screenshot OCR is only a reading aid. Always verify every strike, premium and OI value against Groww. No broker login, live data connection or order placement is included. Hourly comparison needs each snapshot to be uploaded and checked; this version does not fetch Groww data automatically.")
