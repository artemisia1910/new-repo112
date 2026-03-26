"""
ReWeave Sustainable Brand — Synthetic Survey Data Generator
2000 respondents | 36 survey columns | 4 derived target columns
Persona-driven distributions | Realistic Indian audience | Noise + Outliers
"""

import pandas as pd
import numpy as np
import os


def generate_reweave_data(n=2000, output_path="reweave_survey_data.csv"):
    np.random.seed(42)

    PERSONAS = [
        "Eco Evangelist",
        "Aesthetic Minimalist",
        "Deal-Driven Pragmatist",
        "Conscious Gifter",
        "Corporate Buyer",
    ]
    PERSONA_PROBS = [0.22, 0.25, 0.20, 0.18, 0.15]
    personas = np.random.choice(PERSONAS, size=n, p=PERSONA_PROBS)

    rows = []

    for i, persona in enumerate(personas):
        row = {"resp_id": f"resp_{i+1:04d}", "customer_segment": persona}

        # ---------- SECTION A: DEMOGRAPHICS ----------
        age_opts = ["18-24", "25-34", "35-44", "45-54", "55+"]
        age_p = {
            "Eco Evangelist":        [0.18, 0.38, 0.28, 0.12, 0.04],
            "Aesthetic Minimalist":  [0.25, 0.42, 0.22, 0.08, 0.03],
            "Deal-Driven Pragmatist":[0.14, 0.28, 0.32, 0.18, 0.08],
            "Conscious Gifter":      [0.08, 0.22, 0.38, 0.22, 0.10],
            "Corporate Buyer":       [0.04, 0.28, 0.42, 0.20, 0.06],
        }
        row["q1_age_group"] = np.random.choice(age_opts, p=age_p[persona])

        gen_opts = ["Female", "Male", "Non-binary", "Prefer not to say"]
        gen_p = {
            "Eco Evangelist":        [0.58, 0.38, 0.03, 0.01],
            "Aesthetic Minimalist":  [0.62, 0.34, 0.03, 0.01],
            "Deal-Driven Pragmatist":[0.45, 0.52, 0.02, 0.01],
            "Conscious Gifter":      [0.55, 0.42, 0.02, 0.01],
            "Corporate Buyer":       [0.38, 0.58, 0.02, 0.02],
        }
        row["q2_gender"] = np.random.choice(gen_opts, p=gen_p[persona])

        reg_opts = ["North India", "South India", "East India", "West India", "Central India", "Northeast India"]
        reg_p = {
            "Eco Evangelist":        [0.25, 0.26, 0.14, 0.24, 0.07, 0.04],
            "Aesthetic Minimalist":  [0.26, 0.25, 0.13, 0.26, 0.07, 0.03],
            "Deal-Driven Pragmatist":[0.30, 0.20, 0.16, 0.18, 0.11, 0.05],
            "Conscious Gifter":      [0.28, 0.22, 0.15, 0.22, 0.08, 0.05],
            "Corporate Buyer":       [0.24, 0.28, 0.12, 0.26, 0.06, 0.04],
        }
        row["q3_region"] = np.random.choice(reg_opts, p=reg_p[persona])

        city_opts = ["Metro/Tier-1", "Tier-2", "Tier-3", "Town/Rural"]
        city_p = {
            "Eco Evangelist":        [0.55, 0.30, 0.12, 0.03],
            "Aesthetic Minimalist":  [0.50, 0.35, 0.12, 0.03],
            "Deal-Driven Pragmatist":[0.28, 0.38, 0.25, 0.09],
            "Conscious Gifter":      [0.38, 0.35, 0.20, 0.07],
            "Corporate Buyer":       [0.70, 0.22, 0.06, 0.02],
        }
        row["q4_city_tier"] = np.random.choice(city_opts, p=city_p[persona])

        occ_opts = ["Student", "Salaried-Private", "Salaried-Govt", "Business Owner", "Homemaker", "Freelancer", "Retired"]
        occ_p = {
            "Eco Evangelist":        [0.15, 0.35, 0.12, 0.18, 0.08, 0.10, 0.02],
            "Aesthetic Minimalist":  [0.20, 0.38, 0.08, 0.14, 0.08, 0.10, 0.02],
            "Deal-Driven Pragmatist":[0.12, 0.30, 0.15, 0.20, 0.14, 0.06, 0.03],
            "Conscious Gifter":      [0.08, 0.28, 0.14, 0.22, 0.18, 0.06, 0.04],
            "Corporate Buyer":       [0.02, 0.42, 0.10, 0.35, 0.04, 0.06, 0.01],
        }
        row["q5_occupation"] = np.random.choice(occ_opts, p=occ_p[persona])

        # ---------- SECTION B: INCOME & SPENDING ----------
        inc_opts = ["Below 25K", "25K-50K", "50K-1L", "1L-2L", "Above 2L"]
        inc_p = {
            "Eco Evangelist":        [0.08, 0.22, 0.38, 0.22, 0.10],
            "Aesthetic Minimalist":  [0.12, 0.28, 0.36, 0.18, 0.06],
            "Deal-Driven Pragmatist":[0.28, 0.38, 0.24, 0.08, 0.02],
            "Conscious Gifter":      [0.15, 0.30, 0.32, 0.16, 0.07],
            "Corporate Buyer":       [0.02, 0.08, 0.22, 0.38, 0.30],
        }
        row["q6_income"] = np.random.choice(inc_opts, p=inc_p[persona])

        ls_opts = ["Below 2K", "2K-5K", "5K-15K", "15K-30K", "Above 30K"]
        ls_p = {
            "Eco Evangelist":        [0.08, 0.22, 0.40, 0.22, 0.08],
            "Aesthetic Minimalist":  [0.08, 0.20, 0.38, 0.25, 0.09],
            "Deal-Driven Pragmatist":[0.30, 0.38, 0.22, 0.08, 0.02],
            "Conscious Gifter":      [0.15, 0.32, 0.34, 0.14, 0.05],
            "Corporate Buyer":       [0.02, 0.08, 0.25, 0.38, 0.27],
        }
        row["q7_lifestyle_spend"] = np.random.choice(ls_opts, p=ls_p[persona])

        emi_opts = ["Yes comfortably", "Yes 0% only", "Occasionally", "No upfront"]
        emi_p = {
            "Eco Evangelist":        [0.22, 0.18, 0.28, 0.32],
            "Aesthetic Minimalist":  [0.18, 0.22, 0.30, 0.30],
            "Deal-Driven Pragmatist":[0.15, 0.35, 0.28, 0.22],
            "Conscious Gifter":      [0.20, 0.20, 0.30, 0.30],
            "Corporate Buyer":       [0.42, 0.20, 0.22, 0.16],
        }
        row["q8_emi_pref"] = np.random.choice(emi_opts, p=emi_p[persona])

        # ---------- SECTION C: ECO MINDSET ----------
        q9_means = {"Eco Evangelist": 4.3, "Aesthetic Minimalist": 3.2,
                    "Deal-Driven Pragmatist": 2.1, "Conscious Gifter": 3.5, "Corporate Buyer": 3.8}
        row["q9_eco_importance"] = int(np.clip(round(np.random.normal(q9_means[persona], 0.7)), 1, 5))

        eco_act_opts = ["Reusable bags", "Avoid plastic", "Buy thrifted", "Eco brands", "Donate clothes", "Compost"]
        eco_act_p = {
            "Eco Evangelist":        [0.85, 0.80, 0.45, 0.70, 0.65, 0.40],
            "Aesthetic Minimalist":  [0.60, 0.55, 0.30, 0.40, 0.45, 0.20],
            "Deal-Driven Pragmatist":[0.35, 0.30, 0.20, 0.15, 0.30, 0.10],
            "Conscious Gifter":      [0.65, 0.55, 0.35, 0.45, 0.60, 0.25],
            "Corporate Buyer":       [0.55, 0.50, 0.20, 0.50, 0.40, 0.20],
        }
        sel = [o for o, p in zip(eco_act_opts, eco_act_p[persona]) if np.random.random() < p]
        row["q10_eco_actions"] = "|".join(sel) if sel else "None"

        shop_opts = ["Values buyer", "Design first", "Deal hunter", "Gifter", "Functionalist"]
        shop_p = {
            "Eco Evangelist":        [0.65, 0.15, 0.05, 0.10, 0.05],
            "Aesthetic Minimalist":  [0.10, 0.65, 0.10, 0.08, 0.07],
            "Deal-Driven Pragmatist":[0.05, 0.12, 0.68, 0.08, 0.07],
            "Conscious Gifter":      [0.12, 0.15, 0.08, 0.58, 0.07],
            "Corporate Buyer":       [0.20, 0.15, 0.08, 0.15, 0.42],
        }
        row["q11_shopping_personality"] = np.random.choice(shop_opts, p=shop_p[persona])

        ep_opts = ["Yes always", "Yes if design good", "Maybe", "No"]
        ep_p = {
            "Eco Evangelist":        [0.55, 0.30, 0.12, 0.03],
            "Aesthetic Minimalist":  [0.15, 0.50, 0.28, 0.07],
            "Deal-Driven Pragmatist":[0.05, 0.15, 0.38, 0.42],
            "Conscious Gifter":      [0.22, 0.40, 0.28, 0.10],
            "Corporate Buyer":       [0.35, 0.40, 0.18, 0.07],
        }
        row["q12_eco_premium"] = np.random.choice(ep_opts, p=ep_p[persona])

        # ---------- SECTION D: PRODUCT PREFERENCES ----------
        prod_opts = ["Bags", "Cushion covers", "Scarves", "Stationery", "Kids products", "Quilts", "Wallets"]
        prod_p = {
            "Eco Evangelist":        [0.82, 0.72, 0.55, 0.48, 0.38, 0.60, 0.50],
            "Aesthetic Minimalist":  [0.78, 0.80, 0.60, 0.42, 0.30, 0.65, 0.55],
            "Deal-Driven Pragmatist":[0.55, 0.45, 0.35, 0.30, 0.28, 0.38, 0.40],
            "Conscious Gifter":      [0.65, 0.75, 0.60, 0.55, 0.68, 0.58, 0.62],
            "Corporate Buyer":       [0.45, 0.52, 0.38, 0.78, 0.32, 0.40, 0.45],
        }
        sel_prods = [o for o, p in zip(prod_opts, prod_p[persona]) if np.random.random() < p]
        if not sel_prods:
            sel_prods = ["Bags"]
        row["q13_products"] = "|".join(sel_prods)
        for prod in prod_opts:
            row["arm_" + prod.lower().replace(" ", "_")] = 1 if prod in sel_prods else 0

        bag_opts = ["Large tote", "Laptop bag", "Sling bag", "Clutch", "Backpack", "Drawstring"]
        bag_p = {
            "Eco Evangelist":        [0.70, 0.45, 0.50, 0.30, 0.40, 0.35],
            "Aesthetic Minimalist":  [0.65, 0.40, 0.60, 0.50, 0.35, 0.28],
            "Deal-Driven Pragmatist":[0.55, 0.35, 0.40, 0.25, 0.38, 0.30],
            "Conscious Gifter":      [0.60, 0.30, 0.45, 0.55, 0.28, 0.32],
            "Corporate Buyer":       [0.40, 0.72, 0.35, 0.28, 0.30, 0.20],
        }
        sel_bags = [o for o, p in zip(bag_opts, bag_p[persona]) if np.random.random() < p]
        row["q14_bag_styles"] = "|".join(sel_bags) if sel_bags else "Large tote"

        fab_opts = ["Denim/canvas", "Cotton/khadi", "Silk/satin", "Patchwork", "Jute", "Velvet"]
        fab_p = {
            "Eco Evangelist":        [0.55, 0.78, 0.35, 0.65, 0.60, 0.30],
            "Aesthetic Minimalist":  [0.42, 0.55, 0.65, 0.70, 0.35, 0.55],
            "Deal-Driven Pragmatist":[0.55, 0.60, 0.28, 0.35, 0.40, 0.22],
            "Conscious Gifter":      [0.45, 0.65, 0.50, 0.60, 0.45, 0.40],
            "Corporate Buyer":       [0.50, 0.48, 0.55, 0.45, 0.38, 0.45],
        }
        sel_fab = [o for o, p in zip(fab_opts, fab_p[persona]) if np.random.random() < p]
        row["q15_fabric_pref"] = "|".join(sel_fab) if sel_fab else "Cotton/khadi"

        col_opts = ["Earthy neutrals", "Bright and bold", "Pastels", "Monochrome", "Deep jewel tones", "Multicolour"]
        col_p = {
            "Eco Evangelist":        [0.72, 0.45, 0.38, 0.35, 0.50, 0.55],
            "Aesthetic Minimalist":  [0.60, 0.40, 0.55, 0.58, 0.52, 0.42],
            "Deal-Driven Pragmatist":[0.45, 0.55, 0.40, 0.38, 0.42, 0.50],
            "Conscious Gifter":      [0.50, 0.60, 0.55, 0.35, 0.60, 0.65],
            "Corporate Buyer":       [0.55, 0.35, 0.28, 0.55, 0.58, 0.30],
        }
        sel_col = [o for o, p in zip(col_opts, col_p[persona]) if np.random.random() < p]
        row["q16_colour_pref"] = "|".join(sel_col) if sel_col else "Earthy neutrals"

        occ2_opts = ["Diwali/festive", "Weddings", "Birthdays", "Corporate gifting", "Home renovation", "No occasion", "Travel"]
        occ2_p = {
            "Eco Evangelist":        [0.72, 0.55, 0.65, 0.45, 0.55, 0.30, 0.40],
            "Aesthetic Minimalist":  [0.60, 0.50, 0.60, 0.35, 0.65, 0.28, 0.45],
            "Deal-Driven Pragmatist":[0.65, 0.45, 0.55, 0.30, 0.40, 0.45, 0.35],
            "Conscious Gifter":      [0.80, 0.82, 0.85, 0.55, 0.40, 0.15, 0.38],
            "Corporate Buyer":       [0.55, 0.45, 0.42, 0.88, 0.45, 0.25, 0.30],
        }
        sel_occ = [o for o, p in zip(occ2_opts, occ2_p[persona]) if np.random.random() < p]
        row["q17_occasions"] = "|".join(sel_occ) if sel_occ else "No occasion"

        dec_opts = ["Cushion covers", "Table runners", "Wall hangings", "Throws", "Lampshades", "Storage"]
        dec_p = {
            "Eco Evangelist":        [0.72, 0.58, 0.55, 0.60, 0.45, 0.50],
            "Aesthetic Minimalist":  [0.78, 0.65, 0.70, 0.68, 0.55, 0.45],
            "Deal-Driven Pragmatist":[0.45, 0.38, 0.35, 0.40, 0.28, 0.42],
            "Conscious Gifter":      [0.68, 0.58, 0.52, 0.62, 0.45, 0.48],
            "Corporate Buyer":       [0.42, 0.45, 0.40, 0.35, 0.38, 0.55],
        }
        sel_dec = [o for o, p in zip(dec_opts, dec_p[persona]) if np.random.random() < p]
        row["q18_home_decor"] = "|".join(sel_dec) if sel_dec else "Cushion covers"

        # ---------- SECTION E: BRAND & CHANNEL ----------
        barr_opts = ["Higher price", "Quality concern", "No touch-feel", "Limited variety", "No barriers"]
        barr_p = {
            "Eco Evangelist":        [0.10, 0.15, 0.20, 0.10, 0.45],
            "Aesthetic Minimalist":  [0.15, 0.25, 0.30, 0.15, 0.15],
            "Deal-Driven Pragmatist":[0.45, 0.25, 0.18, 0.08, 0.04],
            "Conscious Gifter":      [0.20, 0.22, 0.25, 0.15, 0.18],
            "Corporate Buyer":       [0.08, 0.12, 0.15, 0.12, 0.53],
        }
        row["q19_barrier"] = np.random.choice(barr_opts, p=barr_p[persona])

        disc_opts = ["Instagram", "YouTube", "Pinterest", "Word of mouth", "Amazon/Flipkart", "In-store", "Influencers"]
        disc_p = {
            "Eco Evangelist":        [0.72, 0.55, 0.55, 0.65, 0.45, 0.35, 0.50],
            "Aesthetic Minimalist":  [0.80, 0.48, 0.70, 0.55, 0.42, 0.38, 0.62],
            "Deal-Driven Pragmatist":[0.45, 0.38, 0.28, 0.55, 0.70, 0.45, 0.35],
            "Conscious Gifter":      [0.58, 0.42, 0.45, 0.72, 0.50, 0.48, 0.42],
            "Corporate Buyer":       [0.38, 0.32, 0.28, 0.58, 0.35, 0.28, 0.25],
        }
        sel_disc = [o for o, p in zip(disc_opts, disc_p[persona]) if np.random.random() < p]
        row["q20_discovery"] = "|".join(sel_disc) if sel_disc else "Instagram"

        brand_opts = ["FabIndia", "Nicobar", "Chumbak", "Pepperfry", "Doodlage", "Local exhibitions", "Meesho/budget"]
        brand_p = {
            "Eco Evangelist":        [0.58, 0.35, 0.30, 0.32, 0.45, 0.50, 0.20],
            "Aesthetic Minimalist":  [0.52, 0.45, 0.42, 0.40, 0.25, 0.38, 0.22],
            "Deal-Driven Pragmatist":[0.28, 0.15, 0.25, 0.35, 0.15, 0.42, 0.65],
            "Conscious Gifter":      [0.50, 0.28, 0.35, 0.40, 0.25, 0.58, 0.30],
            "Corporate Buyer":       [0.55, 0.45, 0.25, 0.50, 0.22, 0.38, 0.15],
        }
        sel_brand = [o for o, p in zip(brand_opts, brand_p[persona]) if np.random.random() < p]
        row["q21_existing_brands"] = "|".join(sel_brand) if sel_brand else "None"

        # ---------- SECTION F: WILLINGNESS TO SPEND ----------
        bsp_opts = ["Below 500", "500-1500", "1500-3500", "3500-7000", "Above 7000"]
        bsp_p = {
            "Eco Evangelist":        [0.05, 0.22, 0.40, 0.25, 0.08],
            "Aesthetic Minimalist":  [0.05, 0.18, 0.38, 0.28, 0.11],
            "Deal-Driven Pragmatist":[0.30, 0.42, 0.22, 0.05, 0.01],
            "Conscious Gifter":      [0.12, 0.30, 0.38, 0.15, 0.05],
            "Corporate Buyer":       [0.02, 0.08, 0.25, 0.38, 0.27],
        }
        row["q22_bag_spend"] = np.random.choice(bsp_opts, p=bsp_p[persona])

        dsp_opts = ["Below 800", "800-2500", "2500-6000", "6000-15000", "Above 15000"]
        dsp_p = {
            "Eco Evangelist":        [0.05, 0.22, 0.42, 0.22, 0.09],
            "Aesthetic Minimalist":  [0.04, 0.18, 0.40, 0.28, 0.10],
            "Deal-Driven Pragmatist":[0.30, 0.42, 0.22, 0.05, 0.01],
            "Conscious Gifter":      [0.10, 0.28, 0.40, 0.17, 0.05],
            "Corporate Buyer":       [0.02, 0.08, 0.22, 0.40, 0.28],
        }
        row["q23_decor_spend"] = np.random.choice(dsp_opts, p=dsp_p[persona])

        sub_opts = ["Yes definitely", "Yes if quality", "Maybe trial first", "No"]
        sub_p = {
            "Eco Evangelist":        [0.40, 0.32, 0.20, 0.08],
            "Aesthetic Minimalist":  [0.22, 0.38, 0.28, 0.12],
            "Deal-Driven Pragmatist":[0.08, 0.18, 0.28, 0.46],
            "Conscious Gifter":      [0.28, 0.35, 0.25, 0.12],
            "Corporate Buyer":       [0.35, 0.32, 0.22, 0.11],
        }
        row["q24_subscription"] = np.random.choice(sub_opts, p=sub_p[persona])

        # ---------- SECTION G: CLASSIFICATION TARGET ----------
        intent_opts = ["Yes", "Maybe", "No"]
        intent_p = {
            "Eco Evangelist":        [0.72, 0.22, 0.06],
            "Aesthetic Minimalist":  [0.35, 0.52, 0.13],
            "Deal-Driven Pragmatist":[0.12, 0.45, 0.43],
            "Conscious Gifter":      [0.52, 0.38, 0.10],
            "Corporate Buyer":       [0.65, 0.28, 0.07],
        }
        row["q25_purchase_intent"] = np.random.choice(intent_opts, p=intent_p[persona])

        # ---------- GAP QUESTIONS (QA - QK) ----------
        qa_opts = ["More excited", "No effect", "Slightly less interested", "Much more - would share"]
        qa_p = {
            "Eco Evangelist":        [0.38, 0.18, 0.04, 0.40],
            "Aesthetic Minimalist":  [0.35, 0.40, 0.15, 0.10],
            "Deal-Driven Pragmatist":[0.18, 0.38, 0.35, 0.09],
            "Conscious Gifter":      [0.40, 0.30, 0.12, 0.18],
            "Corporate Buyer":       [0.35, 0.40, 0.10, 0.15],
        }
        row["qa_quality_perception"] = np.random.choice(qa_opts, p=qa_p[persona])

        qb_opts = ["See physically", "Photos and reviews", "Certification", "Recommendation", "Artisan video"]
        qb_p = {
            "Eco Evangelist":        [0.18, 0.22, 0.35, 0.12, 0.13],
            "Aesthetic Minimalist":  [0.32, 0.35, 0.15, 0.10, 0.08],
            "Deal-Driven Pragmatist":[0.38, 0.32, 0.12, 0.12, 0.06],
            "Conscious Gifter":      [0.28, 0.28, 0.18, 0.18, 0.08],
            "Corporate Buyer":       [0.15, 0.20, 0.45, 0.12, 0.08],
        }
        row["qb_quality_proof"] = np.random.choice(qb_opts, p=qb_p[persona])

        qc_opts = ["More excited to buy", "No effect", "Slightly less interested", "Much more - would share"]
        qc_p = {
            "Eco Evangelist":        [0.38, 0.12, 0.02, 0.48],
            "Aesthetic Minimalist":  [0.30, 0.45, 0.18, 0.07],
            "Deal-Driven Pragmatist":[0.15, 0.42, 0.35, 0.08],
            "Conscious Gifter":      [0.42, 0.28, 0.08, 0.22],
            "Corporate Buyer":       [0.38, 0.35, 0.08, 0.19],
        }
        row["qc_narrative_resonance"] = np.random.choice(qc_opts, p=qc_p[persona])

        qd_means = {"Eco Evangelist": 4.5, "Aesthetic Minimalist": 3.0,
                    "Deal-Driven Pragmatist": 2.2, "Conscious Gifter": 3.5, "Corporate Buyer": 3.8}
        row["qd_traceability"] = int(np.clip(round(np.random.normal(qd_means[persona], 0.8)), 1, 5))

        qe_opts = ["Almost always", "Often", "Sometimes", "Rarely", "Dont pay attention"]
        qe_p = {
            "Eco Evangelist":        [0.08, 0.22, 0.40, 0.25, 0.05],
            "Aesthetic Minimalist":  [0.15, 0.30, 0.38, 0.12, 0.05],
            "Deal-Driven Pragmatist":[0.38, 0.35, 0.18, 0.05, 0.04],
            "Conscious Gifter":      [0.12, 0.28, 0.38, 0.16, 0.06],
            "Corporate Buyer":       [0.10, 0.22, 0.40, 0.22, 0.06],
        }
        row["qe_greenwashing"] = np.random.choice(qe_opts, p=qe_p[persona])

        qf_opts = ["Yes pay more", "Yes no extra cost", "No prefer ready-made", "Maybe for gift"]
        qf_p = {
            "Eco Evangelist":        [0.38, 0.28, 0.18, 0.16],
            "Aesthetic Minimalist":  [0.42, 0.30, 0.18, 0.10],
            "Deal-Driven Pragmatist":[0.12, 0.30, 0.42, 0.16],
            "Conscious Gifter":      [0.28, 0.25, 0.15, 0.32],
            "Corporate Buyer":       [0.45, 0.28, 0.15, 0.12],
        }
        row["qf_customisation"] = np.random.choice(qf_opts, p=qf_p[persona])

        qg_means = {"Eco Evangelist": 8.2, "Aesthetic Minimalist": 6.8,
                    "Deal-Driven Pragmatist": 5.2, "Conscious Gifter": 7.5, "Corporate Buyer": 7.8}
        qg = int(np.clip(round(np.random.normal(qg_means[persona], 1.5)), 1, 10))
        row["qg_nps_score"] = qg
        row["nps_segment"] = "Promoter" if qg >= 9 else ("Passive" if qg >= 7 else "Detractor")

        qh_opts = ["Yes regularly", "Yes once or twice", "No but would", "No dont share"]
        qh_p = {
            "Eco Evangelist":        [0.42, 0.32, 0.18, 0.08],
            "Aesthetic Minimalist":  [0.38, 0.35, 0.18, 0.09],
            "Deal-Driven Pragmatist":[0.12, 0.25, 0.28, 0.35],
            "Conscious Gifter":      [0.25, 0.38, 0.25, 0.12],
            "Corporate Buyer":       [0.15, 0.25, 0.28, 0.32],
        }
        row["qh_social_sharing"] = np.random.choice(qh_opts, p=qh_p[persona])

        qi_opts = ["Yes decision-maker", "Yes influence", "No", "Self-employed"]
        qi_p = {
            "Eco Evangelist":        [0.08, 0.18, 0.62, 0.12],
            "Aesthetic Minimalist":  [0.06, 0.14, 0.68, 0.12],
            "Deal-Driven Pragmatist":[0.08, 0.12, 0.65, 0.15],
            "Conscious Gifter":      [0.12, 0.20, 0.55, 0.13],
            "Corporate Buyer":       [0.68, 0.18, 0.08, 0.06],
        }
        row["qi_b2b_role"] = np.random.choice(qi_opts, p=qi_p[persona])

        qj_opts = ["No design matters", "Yes up to 20%", "Yes up to 40%", "Yes significantly more"]
        qj_p = {
            "Eco Evangelist":        [0.08, 0.28, 0.35, 0.29],
            "Aesthetic Minimalist":  [0.15, 0.38, 0.32, 0.15],
            "Deal-Driven Pragmatist":[0.48, 0.30, 0.16, 0.06],
            "Conscious Gifter":      [0.12, 0.35, 0.32, 0.21],
            "Corporate Buyer":       [0.10, 0.25, 0.35, 0.30],
        }
        row["qj_artisan_premium"] = np.random.choice(qj_opts, p=qj_p[persona])

        qk_opts = ["Very comfortable D2C", "Prefer marketplaces", "Known brands only", "Prefer offline"]
        qk_p = {
            "Eco Evangelist":        [0.52, 0.28, 0.15, 0.05],
            "Aesthetic Minimalist":  [0.48, 0.32, 0.15, 0.05],
            "Deal-Driven Pragmatist":[0.22, 0.48, 0.18, 0.12],
            "Conscious Gifter":      [0.35, 0.38, 0.18, 0.09],
            "Corporate Buyer":       [0.55, 0.28, 0.12, 0.05],
        }
        row["qk_d2c_comfort"] = np.random.choice(qk_opts, p=qk_p[persona])

        rows.append(row)

    df = pd.DataFrame(rows)

    # ===================================
    # DERIVE SPENDING SCORE (0-100)
    # ===================================
    income_v = {"Below 25K": 1, "25K-50K": 2, "50K-1L": 3, "1L-2L": 4, "Above 2L": 5}
    lifestyle_v = {"Below 2K": 1, "2K-5K": 2, "5K-15K": 3, "15K-30K": 4, "Above 30K": 5}
    bag_v = {"Below 500": 1, "500-1500": 2, "1500-3500": 3, "3500-7000": 4, "Above 7000": 5}
    dec_v = {"Below 800": 1, "800-2500": 2, "2500-6000": 3, "6000-15000": 4, "Above 15000": 5}
    art_v = {"No design matters": 0, "Yes up to 20%": 3, "Yes up to 40%": 6, "Yes significantly more": 10}

    inc_s = df["q6_income"].map(income_v).fillna(2) * 8
    ls_s = df["q7_lifestyle_spend"].map(lifestyle_v).fillna(2) * 4
    bag_s = df["q22_bag_spend"].map(bag_v).fillna(2) * 3
    dec_s = df["q23_decor_spend"].map(dec_v).fillna(2) * 3
    art_s = df["qj_artisan_premium"].map(art_v).fillna(0)

    base = (inc_s + ls_s + bag_s + dec_s + art_s).astype(float)

    b2b_mask = df["qi_b2b_role"] == "Yes decision-maker"
    mult = np.where(b2b_mask, np.random.uniform(2.0, 4.0, len(df)), 1.0)
    base = base * mult

    noise = np.random.normal(0, 3, len(df))
    df["spending_score"] = np.clip(base + noise, 0, 100).round(2)
    df["purchase_intent_encoded"] = df["q25_purchase_intent"].map({"Yes": 2, "Maybe": 1, "No": 0})

    # ===================================
    # INJECT NOISE & OUTLIERS
    # ===================================
    high_eco = df[df["q9_eco_importance"] >= 4].index.tolist()
    contradict = np.random.choice(high_eco, size=min(150, len(high_eco)), replace=False)
    flip = contradict[: len(contradict) // 2]
    df.loc[flip, "q25_purchase_intent"] = "No"
    df.loc[flip, "purchase_intent_encoded"] = 0

    low_inc = df[df["q6_income"] == "Below 25K"].index.tolist()
    aspirational = np.random.choice(low_inc, size=min(80, len(low_inc)), replace=False)
    df.loc[aspirational, "q22_bag_spend"] = np.random.choice(["3500-7000", "Above 7000"], len(aspirational))

    straight = np.random.choice(df.index, size=40, replace=False)
    df.loc[straight, "q9_eco_importance"] = np.random.choice([1, 5], len(straight))
    df.loc[straight, "qd_traceability"] = df.loc[straight, "q9_eco_importance"]

    b2b_top = df[df["qi_b2b_role"] == "Yes decision-maker"].index[:60]
    df.loc[b2b_top, "spending_score"] = np.random.uniform(85, 100, len(b2b_top))

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["resp_id"] = [f"resp_{i+1:04d}" for i in range(len(df))]

    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path} | Rows: {len(df)} | Columns: {len(df.columns)}")
    return df


if __name__ == "__main__":
    generate_reweave_data()
