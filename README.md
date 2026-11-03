# MCX Natural Gas Position Risk Monitor v3

This Streamlit prototype is tuned to the Groww option-chain screenshot layout shown by the user: **Call OI | Call price | Strike | Put price | Put OI**.

## Run

1. Install Python 3.10 or later.
2. Extract this ZIP.
3. Open a terminal in `mcx_ng_risk_monitor`.
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.
6. In the browser, upload a Groww screenshot and click **Extract option chain from screenshot**.
7. Verify every extracted number in the editable table. Enter the current futures price and your short CE/PE positions, then run analysis.

## Screenshot support and limitations

- Supports a clear screenshot where the five columns above and strike labels are visible.
- The parser reads visible OI and option prices. It does not infer IV, volume, delta, gamma, or missing rows outside the screenshot.
- If the screenshot is cropped, blurry, or the app layout changes, OCR may fail or misread digits. Always verify the table.
- The underlying/futures price, expiry and your position details are entered separately.
- Uploading a new screenshot does not automatically establish the change from a prior snapshot unless you retain and compare the previous data.

## Position inputs

For each open short option, enter side (PE/CE), strike, original selling price, current LTP, total quantity in units, and expiry. The app estimates premium P&L as `(sell price - current LTP) × quantity`, before charges.

## Market analysis

For a directional comparison of put-selling versus call-selling risk, upload futures history CSV with hourly OHLCV and OI data. Without sufficient futures history, the app intentionally avoids claiming one side is safer. PCR from the visible uploaded strikes is only a partial-chain PCR, not full-market PCR.

## Data safety

No Groww login, broker connection, live data feed, order placement, or automatic background data fetching is included. Screenshots are uploaded to the Streamlit app you run/deploy. Do not share broker passwords or OTPs.
