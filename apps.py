import streamlit as st
import pandas as pd
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------------------
# Load trained model + encoder
# ---------------------------
MODEL = joblib.load("carbon_model.pkl")
LE = joblib.load("label_encoder.pkl")

# ---------------------------
# Load supply chain data
# ---------------------------
sc_df = pd.read_csv("carbon_footprint_supply_chain.csv")

# Standardize column names
sc_df.columns = sc_df.columns.str.strip().str.lower()

# Rename required columns
sc_df.rename(columns={
    "shipment_id": "shipment_id",
    "origin": "origin",
    "destination": "destination",
    "distance_km": "distance_km",
    "transport_mode": "mode",
    "fuel_type": "fuel_type",
    "load_utilization_%": "load_utilization",
    "fuel_consumed_l": "fuel_consumed_l",
    "co2_emissions_kg": "emissions_kgco2e",
    "shipment_weight_kg": "weight_kg",   
    "delivery_time_hr": "delivery_time_hr",
    "cost_usd": "cost_usd"
}, inplace=True)

# Convert weight to tons
sc_df["weight_tons"] = sc_df["weight_kg"] / 1000.0

# ---------------------------
# Build graph for shortest path
# ---------------------------
G = nx.DiGraph()
for _, row in sc_df.iterrows():
    G.add_edge(
        row["origin"],
        row["destination"],
        distance=row["distance_km"],
        emission=row["emissions_kgco2e"]
    )

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌍 Carbon Footprint Optimization in Supply Chain Logistics 🌍")

st.sidebar.header("Select Feature")
feature = st.sidebar.radio("Choose an option:", [
    "Carbon Emission Calculator", 
    "Shortest Route Finder", 
    "Logistics Travel Planner"
])

# ---------------------------
# 1. Carbon Emission Calculator
# ---------------------------
if feature == "Carbon Emission Calculator":
    st.subheader("Carbon Emission Calculator")

    distance = st.number_input("Distance (km)", min_value=1, value=100)
    weight = st.number_input("Weight (tons)", min_value=1.0, value=10.0)
    mode = st.selectbox("Transport Mode", LE.classes_)

    # Encode mode
    mode_enc = LE.transform([mode])[0]

    # Predict
    pred = MODEL.predict([[distance, weight, mode_enc]])[0]

    st.success(f"✅ Estimated Carbon Emission: {pred:.2f} kg CO₂e")


# ---------------------------
# 2. Shortest Route Finder
# ---------------------------
elif feature == "Shortest Route Finder":
    st.subheader("🚚 Find Shortest Route (by carbon emissions)")

    start = st.selectbox("Select Start Location", sc_df["origin"].unique())
    end = st.selectbox("Select Destination", sc_df["destination"].unique())
    transport_choice = st.selectbox("Filter by Transport Mode (optional)", ["All"] + list(sc_df["mode"].unique()))

    if st.button("Find Route"):
        try:
            # Build filtered graph
            G_filtered = nx.DiGraph()
            for _, row in sc_df.iterrows():
                if transport_choice == "All" or row["mode"] == transport_choice:
                    G_filtered.add_edge(
                        row["origin"],
                        row["destination"],
                        distance=row["distance_km"],
                        emission=row["emissions_kgco2e"]
                    )

            # Find optimal route
            path = nx.shortest_path(G_filtered, source=start, target=end, weight="emission")
            total_emission = sum(G_filtered[path[i]][path[i + 1]]["emission"] for i in range(len(path) - 1))
            total_distance = sum(G_filtered[path[i]][path[i + 1]]["distance"] for i in range(len(path) - 1))

            st.write("📍 **Optimal Route (min CO₂):**", " → ".join(path))
            st.write(f"🛣️ Total Distance: {total_distance:.2f} km")
            st.success(f"🌱 Total Emissions (Shortest Route): {total_emission:,.2f} kg CO₂e")

            # Compare with alternative routes
            max_routes = 5
            all_routes = []
            for idx, route in enumerate(nx.all_simple_paths(G_filtered, source=start, target=end, cutoff=5)):
                if idx >= max_routes:  
                    break
                emission = sum(G_filtered[route[i]][route[i + 1]]["emission"] for i in range(len(route) - 1))
                distance = sum(G_filtered[route[i]][route[i + 1]]["distance"] for i in range(len(route) - 1))
                all_routes.append({"route": " → ".join(route), "emission": emission, "distance": distance})

            if all_routes:
                routes_df = pd.DataFrame(all_routes)

                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(routes_df["route"], routes_df["emission"], color="skyblue")
                ax.set_xlabel("Route (Origin → Destination)")
                ax.set_ylabel("Emissions (kg CO₂e)")
                ax.set_title(f"Carbon Emissions for Routes: {start} → {end}")
                plt.xticks(rotation=30, ha="right")

                st.pyplot(fig)
                st.dataframe(routes_df)
            else:
                st.warning("⚠️ No alternative routes found.")
        except nx.NetworkXNoPath:
            st.error("❌ No available route between selected nodes.")


# ---------------------------
# 3. Logistics Travel Planner
# ---------------------------
elif feature == "Logistics Travel Planner":
    st.subheader("🚛 Plan Multi-Mode Logistics Travel")

    num_legs = st.number_input("How many travel legs?", min_value=1, max_value=5, value=2)

    legs = []
    prev_destination = None
    for i in range(num_legs):
        st.markdown(f"### Leg {i+1}")

        # Auto-origin from previous destination
        if i == 0:
            origin = st.selectbox(f"Origin (Leg {i+1})", sc_df["origin"].unique(), key=f"origin_{i}")
        else:
            origin = prev_destination
            st.text(f"Origin (Leg {i+1}): {origin}")

        destination = st.selectbox(f"Destination (Leg {i+1})", sc_df["destination"].unique(), key=f"dest_{i}")
        mode = st.selectbox(f"Transport Mode (Leg {i+1})", sc_df["mode"].unique(), key=f"mode_{i}")
        weight = st.number_input(f"Weight (tons) (Leg {i+1})", min_value=1.0, value=10.0, key=f"weight_{i}")

        # Auto fetch distance if exists in dataset
        distance_val = sc_df[
            (sc_df["origin"] == origin) & 
            (sc_df["destination"] == destination) & 
            (sc_df["mode"] == mode)
        ]["distance_km"]

        if not distance_val.empty:
            distance = float(distance_val.iloc[0])
            st.text(f"Distance (km): {distance} (auto-fetched)")
        else:
            distance = st.number_input(f"Distance (km) (Leg {i+1})", min_value=1, value=100, key=f"dist_{i}")

        legs.append({"origin": origin, "destination": destination, "mode": mode, "distance": distance, "weight": weight})
        prev_destination = destination  # Set for next leg

    if st.button("Calculate Total Emissions"):
        results = []
        total_emission = 0
        for idx, leg in enumerate(legs):
            mode_enc = LE.transform([leg["mode"]])[0]
            pred = MODEL.predict([[leg["distance"], leg["weight"], mode_enc]])[0]
            total_emission += pred
            results.append({
                "Leg": idx + 1,
                "Origin": leg["origin"],
                "Destination": leg["destination"],
                "Mode": leg["mode"],
                "Distance_km": leg["distance"],
                "Weight_tons": leg["weight"],
                "Emission_kgCO2e": pred
            })

        results_df = pd.DataFrame(results)

        # Show results table
        st.dataframe(results_df)

        # Show total
        st.success(f"🌱 Total Emissions for Planned Trip: {total_emission:,.2f} kg CO₂e")

        # Pie chart of emissions by leg
        fig, ax = plt.subplots()
        ax.pie(results_df["Emission_kgCO2e"], labels=results_df["Leg"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Emission Contribution per Leg")
        st.pyplot(fig)

        # Download as Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="Logistics Plan")
        st.download_button(
            label="📥 Download Logistics Plan as Excel",
            data=buffer.getvalue(),
            file_name="logistics_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
