"""
Calendario Historico de Eventos 2000-2026
==========================================
Fuente: Wikipedia, NBER, Federal Reserve, OPEC, datos publicos.
Cada evento se mapea a uno de los 30 tipos del EVENT_SUBSECTOR_MAP.

Formato: (event_type, start_date, end_date, intensity, label)
- intensity: 0.5 = leve, 1.0 = moderado, 1.5 = fuerte, 2.0 = extremo
- Las fechas definen el PERIODO donde el evento esta activo
"""

EVENT_CALENDAR = [

    # =================================================================
    # TIPOS DE INTERES (Federal Reserve)
    # =================================================================
    # Ciclos de subida
    ("subida_tipos_interes", "2000-02-02", "2000-05-16", 1.5, "Greenspan hikes 6.0->6.5%"),
    ("subida_tipos_interes", "2004-06-30", "2006-06-29", 1.5, "Greenspan/Bernanke 1.0->5.25% (17 hikes)"),
    ("subida_tipos_interes", "2015-12-16", "2018-12-19", 1.0, "Yellen/Powell 0.25->2.50% (9 hikes)"),
    ("subida_tipos_interes", "2022-03-16", "2022-12-31", 2.0, "Powell 0.25->4.50% aggressive 75bp hikes"),
    ("subida_tipos_interes", "2023-01-01", "2023-07-26", 1.0, "Powell deceleration 25bp steps, end in sight"),

    # Ciclos de bajada
    ("bajada_tipos_interes", "2001-01-03", "2001-12-11", 1.5, "Greenspan 6.5->1.75% aggressive 475bp in 11 months"),
    ("bajada_tipos_interes", "2002-01-01", "2003-06-25", 1.0, "Greenspan 1.75->1.0% deceleration, only 75bp in 18 months"),
    ("bajada_tipos_interes", "2007-09-18", "2008-12-16", 2.0, "Bernanke 5.25->0% emergency"),
    ("bajada_tipos_interes", "2019-07-31", "2019-10-30", 1.0, "Powell insurance cuts 2.50->1.75%"),
    ("bajada_tipos_interes", "2020-03-03", "2020-03-15", 2.0, "COVID emergency 1.75->0%"),
    ("bajada_tipos_interes", "2024-09-18", "2025-01-29", 1.0, "Powell cuts 5.50->4.25%"),
    ("bajada_tipos_interes", "2026-06-01", "2026-12-31", 1.0, "Fed expected cuts 4.25->3.0%"),

    # =================================================================
    # GEOPOLITICA: MEDIO ORIENTE
    # =================================================================
    ("guerra_medio_oriente", "2001-10-07", "2002-03-31", 1.5, "Afghanistan invasion post-9/11"),
    ("guerra_medio_oriente", "2003-03-20", "2003-05-01", 2.0, "Iraq War - invasion phase"),
    ("guerra_medio_oriente", "2003-05-01", "2004-12-31", 1.0, "Iraq War - occupation/insurgency"),
    ("guerra_medio_oriente", "2006-07-12", "2006-08-14", 1.5, "Israel-Lebanon War (Hezbollah)"),
    ("guerra_medio_oriente", "2011-03-19", "2011-10-31", 1.0, "Libya NATO intervention"),
    ("guerra_medio_oriente", "2013-08-21", "2013-09-15", 1.5, "Syria chemical weapons crisis"),
    ("guerra_medio_oriente", "2014-06-13", "2014-12-31", 1.5, "ISIS rise in Iraq/Syria"),
    ("guerra_medio_oriente", "2017-04-07", "2017-04-14", 1.0, "US Syria tomahawk strikes"),
    ("guerra_medio_oriente", "2019-09-14", "2019-10-15", 1.5, "Saudi Aramco drone attack"),
    ("guerra_medio_oriente", "2020-01-03", "2020-01-12", 2.0, "US kills Soleimani - Iran crisis"),
    ("guerra_medio_oriente", "2023-10-07", "2024-04-30", 2.0, "Israel-Hamas war (Oct 7)"),
    ("guerra_medio_oriente", "2024-01-12", "2024-06-30", 1.5, "Houthi Red Sea shipping attacks"),
    ("guerra_medio_oriente", "2024-04-13", "2024-04-20", 2.0, "Iran-Israel direct strikes"),
    ("guerra_medio_oriente", "2024-05-01", "2025-06-30", 1.5, "Israel-Hamas war extended + Lebanon ops"),

    # =================================================================
    # GEOPOLITICA: RUSIA / UCRANIA
    # =================================================================
    ("guerra_rusia_ucrania", "2008-08-08", "2008-08-28", 1.5, "Russia-Georgia war"),
    ("guerra_rusia_ucrania", "2014-02-20", "2014-09-05", 1.5, "Crimea annexation + Donbass"),
    ("guerra_rusia_ucrania", "2022-02-24", "2022-06-30", 2.0, "Russia invades Ukraine - peak"),
    ("guerra_rusia_ucrania", "2022-07-01", "2022-12-31", 1.5, "Ukraine war - energy crisis"),
    ("guerra_rusia_ucrania", "2023-01-01", "2025-12-31", 0.8, "Ukraine war ongoing"),
    ("guerra_rusia_ucrania", "2026-01-01", "2026-06-30", 0.5, "Ukraine ceasefire talks Paris/Geneva"),

    # =================================================================
    # GEOPOLITICA: CHINA / TAIWAN
    # =================================================================
    ("tension_china_taiwan", "2010-09-07", "2010-09-24", 1.0, "China-Japan Senkaku crisis"),
    ("tension_china_taiwan", "2018-03-22", "2018-12-31", 1.5, "US-China trade war begins"),
    ("tension_china_taiwan", "2019-05-10", "2019-12-15", 2.0, "US-China trade war escalation"),
    ("tension_china_taiwan", "2020-05-20", "2020-08-31", 1.0, "Hong Kong security law"),
    ("tension_china_taiwan", "2022-08-02", "2022-08-31", 2.0, "Pelosi visits Taiwan"),
    ("tension_china_taiwan", "2022-10-07", "2023-03-31", 1.5, "US chip export ban to China"),
    ("tension_china_taiwan", "2025-02-04", "2025-05-12", 1.5, "Trump tariffs China escalation to 145%"),
    ("tension_china_taiwan", "2025-05-13", "2026-06-30", 1.0, "US-China tariff truce + tech restrictions"),

    # =================================================================
    # SANCIONES COMERCIALES
    # =================================================================
    ("sanciones_comerciales", "2002-03-05", "2003-12-04", 1.0, "Bush steel tariffs (Section 201)"),
    ("sanciones_comerciales", "2018-03-01", "2018-07-06", 1.5, "Trump steel/aluminum tariffs"),
    ("sanciones_comerciales", "2018-07-06", "2020-01-15", 2.0, "US-China tariff war"),
    ("sanciones_comerciales", "2022-02-25", "2022-12-31", 2.0, "Russia sanctions peak impact"),
    ("sanciones_comerciales", "2023-01-01", "2023-12-31", 0.8, "Russia sanctions adapted, reduced impact"),
    ("sanciones_comerciales", "2025-02-01", "2025-04-01", 1.5, "Trump universal tariffs initial"),
    ("sanciones_comerciales", "2025-04-02", "2025-05-12", 2.0, "Liberation Day tariffs 145% China"),
    ("sanciones_comerciales", "2025-05-13", "2025-12-31", 1.5, "US-China tariff truce 30% + EU deal 15%"),
    ("sanciones_comerciales", "2026-01-01", "2026-06-30", 1.5, "Global tariffs + SCOTUS IEEPA ruling Feb"),

    # =================================================================
    # TERRORISMO
    # =================================================================
    ("terrorismo", "2001-09-11", "2001-10-15", 2.0, "September 11 attacks"),
    ("terrorismo", "2004-03-11", "2004-03-25", 1.0, "Madrid train bombings"),
    ("terrorismo", "2005-07-07", "2005-07-21", 1.0, "London 7/7 bombings"),
    ("terrorismo", "2013-04-15", "2013-04-26", 1.0, "Boston Marathon bombing"),
    ("terrorismo", "2015-01-07", "2015-01-14", 1.0, "Charlie Hebdo attack Paris"),
    ("terrorismo", "2015-11-13", "2015-12-15", 1.5, "Paris Bataclan attacks"),
    ("terrorismo", "2016-06-12", "2016-07-31", 1.0, "Orlando + Nice attacks"),

    # =================================================================
    # PETROLEO
    # =================================================================
    # Subidas
    ("precio_petroleo_sube", "2002-12-01", "2003-03-20", 1.5, "Iraq War fear premium"),
    ("precio_petroleo_sube", "2004-01-01", "2006-08-07", 1.5, "Oil supercycle $30->$77"),
    ("precio_petroleo_sube", "2007-01-01", "2008-07-11", 2.0, "Oil supercycle to $147"),
    ("precio_petroleo_sube", "2010-12-01", "2011-04-29", 1.5, "Arab Spring oil fears"),
    ("precio_petroleo_sube", "2017-06-01", "2018-10-03", 1.0, "OPEC cuts + Iran sanctions"),
    ("precio_petroleo_sube", "2021-01-01", "2022-06-14", 2.0, "Post-COVID + Ukraine $130"),

    # Bajadas
    ("precio_petroleo_baja", "2001-09-01", "2001-11-30", 1.0, "9/11 demand shock"),
    ("precio_petroleo_baja", "2008-07-14", "2009-02-12", 2.0, "GFC crash $147->$33"),
    ("precio_petroleo_baja", "2014-06-20", "2016-02-11", 2.0, "Shale glut $107->$26"),
    ("precio_petroleo_baja", "2018-10-03", "2018-12-24", 1.5, "Q4 2018 selloff"),
    ("precio_petroleo_baja", "2020-01-20", "2020-04-28", 2.0, "COVID crash / WTI negative"),

    # =================================================================
    # RECESION / CRECIMIENTO
    # =================================================================
    ("recesion", "2001-03-01", "2001-11-30", 1.5, "Dot-com recession (NBER)"),
    ("recesion", "2007-12-01", "2009-06-30", 2.0, "Great Recession (NBER)"),
    ("recesion", "2020-02-01", "2020-04-30", 2.0, "COVID recession (NBER)"),

    ("crecimiento_economico", "2003-07-01", "2006-12-31", 1.0, "Post dot-com recovery"),
    ("crecimiento_economico", "2010-07-01", "2014-12-31", 1.0, "Post-GFC slow recovery"),
    ("crecimiento_economico", "2017-01-01", "2019-06-30", 1.5, "Tax cuts + low unemployment"),
    ("crecimiento_economico", "2020-07-01", "2021-12-31", 2.0, "Post-COVID V-shape boom"),
    ("crecimiento_economico", "2023-04-01", "2024-12-31", 1.0, "Soft landing, employment resilient"),
    ("crecimiento_economico", "2025-01-01", "2026-06-30", 0.8, "US growth resilient despite tariff drag"),

    # =================================================================
    # CRISIS BANCARIA
    # =================================================================
    ("crisis_bancaria", "2007-08-09", "2007-09-30", 1.5, "BNP Paribas / Northern Rock"),
    ("crisis_bancaria", "2008-03-14", "2008-03-28", 2.0, "Bear Stearns collapse"),
    ("crisis_bancaria", "2008-09-15", "2009-03-09", 2.0, "Lehman / AIG / systemic"),
    ("crisis_bancaria", "2010-04-01", "2012-07-31", 1.5, "European debt crisis (PIIGS)"),
    ("crisis_bancaria", "2023-03-10", "2023-05-01", 2.0, "SVB / Signature / First Republic"),

    # =================================================================
    # INFLACION
    # =================================================================
    ("inflacion_alta", "2007-06-01", "2008-07-31", 1.5, "Pre-GFC commodity inflation"),
    ("inflacion_alta", "2021-03-01", "2022-06-30", 2.0, "Post-COVID CPI surge 9.1%"),
    ("inflacion_alta", "2022-07-01", "2022-12-31", 1.5, "Persistent core inflation"),
    ("inflacion_alta", "2023-01-01", "2023-03-31", 1.0, "CPI declining from 6.4%"),
    ("inflacion_alta", "2023-04-01", "2023-06-30", 0.5, "CPI rapid decline toward 3%"),
    ("inflacion_alta", "2025-04-01", "2025-12-31", 1.0, "Tariff-driven inflation resurgence"),
    ("inflacion_alta", "2026-01-01", "2026-06-30", 1.0, "Tariff inflation peak PCE rising"),

    # =================================================================
    # ESTIMULO FISCAL
    # =================================================================
    ("estimulo_fiscal", "2001-06-07", "2001-07-31", 1.0, "Bush tax rebate checks"),
    ("estimulo_fiscal", "2008-02-13", "2008-06-30", 1.0, "Bush stimulus checks"),
    ("estimulo_fiscal", "2008-10-03", "2009-02-17", 2.0, "TARP $700B + ARRA $831B"),
    ("estimulo_fiscal", "2017-12-22", "2018-06-30", 1.0, "Tax Cuts and Jobs Act"),
    ("estimulo_fiscal", "2020-03-27", "2020-12-27", 2.0, "CARES Act $2.2T + PPP"),
    ("estimulo_fiscal", "2021-01-14", "2021-03-11", 2.0, "American Rescue Plan $1.9T"),
    ("estimulo_fiscal", "2021-11-15", "2022-08-16", 1.5, "Infrastructure + IRA + CHIPS"),

    # =================================================================
    # PANDEMIA
    # =================================================================
    ("pandemia", "2009-04-15", "2009-08-31", 1.0, "H1N1 swine flu"),
    ("pandemia", "2014-08-08", "2014-10-31", 1.0, "Ebola scare (West Africa)"),
    ("pandemia", "2020-01-20", "2020-06-30", 2.0, "COVID-19 first wave + lockdowns"),
    ("pandemia", "2020-10-01", "2021-03-31", 1.5, "COVID winter wave pre-vaccine"),
    ("pandemia", "2021-07-01", "2021-09-30", 1.0, "COVID Delta variant wave"),
    ("pandemia", "2022-01-01", "2022-02-28", 0.8, "COVID Omicron (milder)"),
    ("pandemia", "2025-01-15", "2025-03-31", 0.8, "H5N1 bird flu fears"),

    # =================================================================
    # CHINA DEMANDA / DESACELERACION
    # =================================================================
    ("demanda_china_fuerte", "2003-01-01", "2007-10-31", 1.5, "China WTO boom + infrastructure"),
    ("demanda_china_fuerte", "2009-03-01", "2011-06-30", 2.0, "China 4T yuan stimulus"),
    ("demanda_china_fuerte", "2016-03-01", "2017-12-31", 1.0, "China supply-side reform"),
    ("demanda_china_fuerte", "2020-07-01", "2021-06-30", 1.5, "China first to recover COVID"),

    ("china_desaceleracion", "2011-07-01", "2012-09-30", 1.0, "China hard landing fears"),
    ("china_desaceleracion", "2015-06-12", "2016-02-11", 2.0, "China stock crash + yuan deval"),
    ("china_desaceleracion", "2018-07-01", "2019-01-31", 1.5, "Trade war slowdown"),
    ("china_desaceleracion", "2021-09-01", "2022-12-31", 2.0, "Evergrande + zero-COVID"),
    ("china_desaceleracion", "2023-01-01", "2024-09-30", 1.5, "Deflation + property crisis"),

    # =================================================================
    # INMOBILIARIO
    # =================================================================
    ("boom_inmobiliario", "2003-01-01", "2006-06-30", 2.0, "US housing bubble"),
    ("boom_inmobiliario", "2012-01-01", "2014-12-31", 1.0, "Housing recovery post-GFC"),
    ("boom_inmobiliario", "2020-06-01", "2022-06-30", 2.0, "COVID WFH housing boom"),

    ("crisis_inmobiliaria", "2006-07-01", "2009-03-31", 2.0, "Subprime mortgage crisis"),
    ("crisis_inmobiliaria", "2022-07-01", "2022-12-31", 1.5, "Mortgage rate shock 3->7%"),
    ("crisis_inmobiliaria", "2023-01-01", "2023-03-31", 0.5, "High rates persist but housing stabilizes"),

    # =================================================================
    # CONSUMIDOR
    # =================================================================
    ("confianza_consumidor_alta", "2004-01-01", "2006-12-31", 1.0, "Mid-2000s consumer boom"),
    ("confianza_consumidor_alta", "2017-01-01", "2019-12-31", 1.0, "Trump era consumer high"),
    ("confianza_consumidor_alta", "2021-03-01", "2021-11-30", 1.5, "Post-vaccine reopening surge"),
    ("confianza_consumidor_alta", "2023-07-01", "2024-12-31", 1.0, "Soft landing confidence"),
    ("confianza_consumidor_alta", "2025-01-01", "2026-06-30", 0.8, "Consumer spending resilient"),

    ("crisis_consumo", "2001-09-11", "2002-03-31", 1.5, "Post-9/11 consumer fear"),
    ("crisis_consumo", "2008-06-01", "2009-06-30", 2.0, "GFC consumer collapse"),
    ("crisis_consumo", "2011-07-01", "2011-10-31", 1.0, "US debt downgrade/EU crisis"),
    ("crisis_consumo", "2022-06-01", "2022-12-31", 1.0, "Inflation erodes purchasing power"),

    # =================================================================
    # TECNOLOGIA
    # =================================================================
    ("innovacion_ai", "2022-11-30", "2023-05-23", 1.0, "ChatGPT early adoption phase"),
    ("innovacion_ai", "2023-05-24", "2023-12-31", 2.0, "Post-Nvidia AI infrastructure boom"),
    ("innovacion_ai", "2024-01-01", "2026-12-31", 2.0, "AI infrastructure buildout $475B capex"),

    ("regulacion_tech", "2017-09-01", "2017-11-30", 1.0, "EU Google antitrust fine"),
    ("regulacion_tech", "2019-06-01", "2019-12-31", 1.0, "DOJ/FTC Big Tech probes"),
    ("regulacion_tech", "2020-10-20", "2021-06-30", 1.5, "DOJ sues Google, FTC sues FB"),
    ("regulacion_tech", "2024-01-01", "2024-12-31", 1.0, "EU DMA + DOJ Google remedy"),
    ("regulacion_tech", "2025-01-27", "2025-04-30", 1.0, "DeepSeek AI efficiency shock"),
    ("regulacion_tech", "2026-02-01", "2026-06-30", 1.0, "AI code disruption fears software sector"),

    ("ciberseguridad", "2013-06-06", "2013-07-31", 1.0, "Snowden NSA revelations"),
    ("ciberseguridad", "2017-05-12", "2017-07-31", 1.5, "WannaCry + NotPetya attacks"),
    ("ciberseguridad", "2020-12-13", "2021-03-31", 1.5, "SolarWinds hack"),
    ("ciberseguridad", "2021-05-07", "2021-06-30", 1.0, "Colonial Pipeline ransomware"),

    # =================================================================
    # TRANSICION ENERGETICA
    # =================================================================
    ("transicion_energetica", "2015-12-12", "2016-12-31", 1.0, "Paris Agreement signed"),
    ("transicion_energetica", "2019-09-01", "2020-02-29", 0.8, "Greta / climate activism"),
    ("transicion_energetica", "2021-01-20", "2021-12-31", 1.5, "Biden clean energy push"),
    ("transicion_energetica", "2022-08-16", "2024-12-31", 2.0, "Inflation Reduction Act"),
    ("transicion_energetica", "2025-01-01", "2026-12-31", 1.0, "IRA clean energy deployment ongoing"),

    # =================================================================
    # DESASTRES NATURALES
    # =================================================================
    ("desastre_natural", "2005-08-29", "2005-11-30", 2.0, "Hurricane Katrina"),
    ("desastre_natural", "2008-09-13", "2008-10-15", 1.0, "Hurricane Ike (Gulf refineries)"),
    ("desastre_natural", "2010-04-20", "2010-08-31", 1.5, "Deepwater Horizon oil spill"),
    ("desastre_natural", "2011-03-11", "2011-06-30", 2.0, "Japan earthquake + Fukushima"),
    ("desastre_natural", "2012-10-29", "2012-12-31", 1.0, "Hurricane Sandy"),
    ("desastre_natural", "2017-08-25", "2017-10-31", 1.5, "Harvey + Irma + Maria"),
    ("desastre_natural", "2021-02-13", "2021-02-28", 1.0, "Texas winter storm"),
    ("desastre_natural", "2023-08-08", "2023-08-31", 0.8, "Hawaii wildfires"),
    ("desastre_natural", "2025-01-07", "2025-02-28", 1.5, "LA Palisades wildfires"),

    # =================================================================
    # FDA / PHARMA
    # =================================================================
    ("fda_aprobacion", "2020-11-09", "2020-12-18", 2.0, "Pfizer + Moderna COVID vaccine"),
    ("fda_aprobacion", "2021-02-27", "2021-03-15", 1.5, "J&J vaccine EUA"),
    ("fda_aprobacion", "2023-01-06", "2023-02-28", 1.0, "Lecanemab (Alzheimer) approval"),
    ("fda_aprobacion", "2024-03-08", "2024-04-30", 1.0, "GLP-1 obesity drugs momentum"),

    ("regulacion_pharma", "2019-01-01", "2019-12-31", 1.0, "Drug pricing reform debates"),
    ("regulacion_pharma", "2022-08-16", "2023-12-31", 1.5, "IRA Medicare price negotiation"),

    # =================================================================
    # SUPPLY CHAIN
    # =================================================================
    ("supply_chain_crisis", "2011-03-11", "2011-09-30", 1.5, "Japan earthquake supply break"),
    ("supply_chain_crisis", "2020-03-01", "2020-12-31", 1.5, "COVID factory shutdowns"),
    ("supply_chain_crisis", "2021-01-01", "2022-06-30", 2.0, "Global chip/container crisis"),
    ("supply_chain_crisis", "2021-03-23", "2021-03-29", 1.5, "Suez Canal blocked (Ever Given)"),

    # =================================================================
    # INFRAESTRUCTURA
    # =================================================================
    ("infraestructura", "2009-02-17", "2010-12-31", 1.5, "ARRA infrastructure spending"),
    ("infraestructura", "2021-11-15", "2026-12-31", 2.0, "IIJA $1.2T infrastructure"),

    # =================================================================
    # ACCIDENTES AVIACION
    # =================================================================
    ("accidente_aviacion", "2001-11-12", "2001-11-30", 1.0, "AA587 crash JFK post-9/11"),
    ("accidente_aviacion", "2009-06-01", "2009-06-30", 1.0, "Air France AF447 Atlantic"),
    ("accidente_aviacion", "2014-03-08", "2014-04-30", 1.5, "MH370 disappears"),
    ("accidente_aviacion", "2014-07-17", "2014-08-15", 1.5, "MH17 shot down Ukraine"),
    ("accidente_aviacion", "2018-10-29", "2019-03-10", 1.0, "Boeing 737 MAX Lion Air crash"),
    ("accidente_aviacion", "2019-03-10", "2020-11-18", 2.0, "Boeing 737 MAX grounding worldwide"),
    ("accidente_aviacion", "2024-01-05", "2024-03-31", 1.5, "Boeing 737 MAX door plug blowout"),
    ("accidente_aviacion", "2025-01-29", "2025-03-31", 1.5, "American Eagle crash Reagan airport"),

    # =================================================================
    # RECALL AUTO MASIVO
    # =================================================================
    ("recall_auto_masivo", "2009-10-05", "2011-02-28", 1.5, "Toyota unintended acceleration 9M vehicles"),
    ("recall_auto_masivo", "2014-02-07", "2015-12-31", 2.0, "GM ignition switch recall 30M vehicles"),
    ("recall_auto_masivo", "2014-09-01", "2016-12-31", 2.0, "Takata airbag crisis 19M US vehicles"),
    ("recall_auto_masivo", "2015-09-18", "2017-06-30", 2.0, "VW Dieselgate 11M vehicles $30B"),
    ("recall_auto_masivo", "2018-03-23", "2018-06-30", 1.0, "Tesla autopilot fatal crash"),

    # =================================================================
    # REVOLUCION EV
    # =================================================================
    ("revolucion_ev", "2017-07-28", "2017-12-31", 1.5, "Tesla Model 3 first deliveries"),
    ("revolucion_ev", "2020-09-22", "2021-06-30", 1.5, "Tesla Battery Day + S&P500 inclusion"),
    ("revolucion_ev", "2021-05-26", "2021-09-30", 1.0, "Ford F-150 Lightning: legacy OEM EV pivot"),
    ("revolucion_ev", "2023-01-01", "2023-06-30", 1.0, "IRA EV tax credit US manufacturing"),

    # =================================================================
    # RECALL ALIMENTARIO
    # =================================================================
    ("recall_alimentario", "2006-09-14", "2006-10-15", 1.0, "E.coli spinach Dole"),
    ("recall_alimentario", "2008-09-01", "2009-01-31", 1.5, "Peanut Corp salmonella 9 deaths"),
    ("recall_alimentario", "2015-10-01", "2016-03-31", 1.5, "Chipotle E.coli multi-state"),
    ("recall_alimentario", "2018-04-10", "2018-06-30", 1.0, "Romaine lettuce E.coli recall"),
    ("recall_alimentario", "2022-02-17", "2022-07-31", 1.5, "Abbott baby formula recall shortage"),

    # =================================================================
    # ESCANDALO PHARMA
    # =================================================================
    ("escandalo_pharma", "2004-09-30", "2005-12-31", 2.0, "Merck Vioxx withdrawal 50K cardiac deaths"),
    ("escandalo_pharma", "2014-09-10", "2015-06-30", 1.5, "Turing/Shkreli Daraprim 5000% hike"),
    ("escandalo_pharma", "2015-08-01", "2016-06-30", 1.5, "Valeant price gouging + fraud"),
    ("escandalo_pharma", "2017-01-01", "2021-12-31", 1.5, "Opioid crisis lawsuits $8B+"),

    # =================================================================
    # FRAUDE CONTABLE
    # =================================================================
    ("fraude_contable", "2001-10-16", "2002-12-31", 2.0, "Enron collapse $74B shareholder loss"),
    ("fraude_contable", "2002-06-25", "2003-06-30", 2.0, "WorldCom $11B fraud"),
    ("fraude_contable", "2008-12-11", "2009-06-30", 2.0, "Madoff Ponzi $65B"),
    ("fraude_contable", "2019-06-18", "2020-09-30", 1.5, "Wirecard $2B fraud"),

    # =================================================================
    # ESCANDALO BANCARIO
    # =================================================================
    ("escandalo_bancario", "2012-06-27", "2013-06-30", 1.5, "LIBOR rigging 9 banks $9B fines"),
    ("escandalo_bancario", "2013-09-01", "2014-12-31", 1.0, "JPMorgan London Whale $6B loss"),
    ("escandalo_bancario", "2016-09-08", "2018-06-30", 2.0, "Wells Fargo fake accounts 3.5M"),

    # =================================================================
    # ESCANDALO DATOS / PRIVACIDAD
    # =================================================================
    ("escandalo_datos_privacidad", "2018-03-17", "2018-07-31", 2.0, "Cambridge Analytica / Facebook 87M users"),
    ("escandalo_datos_privacidad", "2021-10-04", "2021-11-30", 1.5, "Facebook Papers Haugen whistleblower"),
    ("escandalo_datos_privacidad", "2022-10-27", "2023-06-30", 1.0, "Twitter/X Musk chaos advertiser exodus"),
    ("escandalo_datos_privacidad", "2023-03-23", "2023-09-30", 1.0, "TikTok Congress hearing ban attempts"),

    # =================================================================
    # CRYPTO COLAPSO
    # =================================================================
    ("crypto_colapso", "2018-01-07", "2018-12-31", 2.0, "Bitcoin $20K->$3K crypto winter"),
    ("crypto_colapso", "2021-05-18", "2021-07-31", 1.5, "China bans crypto mining BTC -50%"),
    ("crypto_colapso", "2022-05-07", "2022-05-31", 2.0, "Luna/TerraUST stablecoin collapse $60B"),
    ("crypto_colapso", "2022-11-08", "2023-01-31", 2.0, "FTX collapse $32B Bankman-Fried"),

    # =================================================================
    # HUELGAS SECTORIALES
    # =================================================================
    ("huelga_sector", "2007-11-05", "2008-02-12", 1.0, "WGA writers strike 100 days"),
    ("huelga_sector", "2019-09-16", "2019-10-25", 1.5, "UAW GM strike 40 days 48K workers"),
    ("huelga_sector", "2023-07-14", "2023-11-17", 1.5, "SAG-AFTRA + WGA double strike"),
    ("huelga_sector", "2023-09-15", "2023-10-30", 2.0, "UAW strike Ford+GM+Stellantis"),
    ("huelga_sector", "2024-10-01", "2024-11-04", 1.5, "Boeing machinists strike 33K workers"),

    # =================================================================
    # ACCIDENTE INDUSTRIAL ENERGIA
    # =================================================================
    ("accidente_industrial_energia", "2005-03-23", "2005-06-30", 1.5, "BP Texas City refinery 15 deaths"),
    ("accidente_industrial_energia", "2010-04-20", "2010-09-19", 2.0, "Deepwater Horizon 87-day spill $65B"),
    ("accidente_industrial_energia", "2019-01-25", "2019-06-30", 1.5, "Vale Brumadinho dam 270 deaths"),

    # =================================================================
    # DISRUPCION MARITIMA
    # =================================================================
    ("disrupcion_maritima", "2021-03-23", "2021-03-29", 1.5, "Suez Canal blocked Ever Given"),
    ("disrupcion_maritima", "2021-09-01", "2022-01-31", 2.0, "Port of LA congestion 100+ ships"),
    ("disrupcion_maritima", "2023-07-01", "2024-01-31", 1.5, "Panama Canal drought 40% capacity"),
    ("disrupcion_maritima", "2024-10-01", "2025-01-31", 1.0, "East Coast port ILA strike"),

    # =================================================================
    # CONTRATOS DEFENSA MAYORES
    # =================================================================
    ("contrato_defensa_mayor", "2001-10-26", "2002-03-31", 2.0, "F-35 JSF contract Lockheed $200B"),
    ("contrato_defensa_mayor", "2022-03-01", "2022-12-31", 2.0, "Ukraine war NATO rearmament surge"),
    ("contrato_defensa_mayor", "2023-01-01", "2024-12-31", 1.5, "AUKUS + Pacific defense buildup"),
    ("contrato_defensa_mayor", "2025-01-01", "2026-12-31", 1.5, "European NATO rearmament Ukraine coalition"),

    # =================================================================
    # EXTRA: ciberseguridad adicional
    # =================================================================
    ("ciberseguridad", "2013-12-18", "2014-02-28", 1.0, "Target breach 40M credit cards"),
    ("ciberseguridad", "2024-07-19", "2024-08-31", 1.5, "CrowdStrike update crash 8.5M PCs"),

    # =================================================================
    # EXTRA: supply chain adicional
    # =================================================================
    ("supply_chain_crisis", "2011-10-01", "2012-06-30", 1.5, "Thailand floods HDD shortage"),
]


# ── HELPER FUNCTIONS ──────────────────────────────────────────

import pandas as pd


def get_active_events(date):
    """Return list of (event_type, intensity, label) active on a given date."""
    if isinstance(date, str):
        date = pd.Timestamp(date)

    active = []
    for evt_type, start, end, intensity, label in EVENT_CALENDAR:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if s <= date <= e:
            active.append((evt_type, intensity, label))
    return active


def build_weekly_events(start='2000-01-01', end='2026-02-28'):
    """Build a DataFrame of weekly event indicators.

    Returns DataFrame indexed by Friday dates with columns:
    - One column per event_type containing intensity (0 if not active)
    """
    dates = pd.date_range(start, end, freq='W-FRI')
    all_event_types = sorted(set(e[0] for e in EVENT_CALENDAR))

    data = []
    for date in dates:
        row = {'date': date}
        # Get all active events for this date
        active = get_active_events(date)
        # For each event type, take the MAX intensity if multiple instances
        type_intensities = {}
        for evt_type, intensity, label in active:
            type_intensities[evt_type] = max(
                type_intensities.get(evt_type, 0), intensity
            )
        for evt_type in all_event_types:
            row[evt_type] = type_intensities.get(evt_type, 0)
        data.append(row)

    df = pd.DataFrame(data).set_index('date')
    return df


def print_summary():
    """Print calendar summary."""
    from collections import Counter

    types = Counter(e[0] for e in EVENT_CALENDAR)
    total_events = len(EVENT_CALENDAR)

    print(f"{'=' * 60}")
    print(f"  Calendario de Eventos Historicos 2000-2026")
    print(f"  {total_events} entradas, {len(types)} tipos de evento")
    print(f"{'=' * 60}")

    for evt_type, count in sorted(types.items(), key=lambda x: -x[1]):
        # Get date range
        entries = [(s, e, i, l) for t, s, e, i, l in EVENT_CALENDAR if t == evt_type]
        first = min(e[0] for e in entries)
        last = max(e[1] for e in entries)
        print(f"  {evt_type:35s}: {count:2d} entries  ({first} -> {last})")

    # Timeline validation
    print(f"\n  Validacion por decada:")
    for decade_start in range(2000, 2026, 5):
        decade_end = decade_start + 5
        count = sum(1 for _, s, e, _, _ in EVENT_CALENDAR
                    if int(s[:4]) < decade_end and int(e[:4]) >= decade_start)
        print(f"    {decade_start}-{decade_end}: {count} eventos activos")

    # Check for known events
    print(f"\n  Validacion fechas clave:")
    checks = [
        ("2001-09-14", "Post 9/11"),
        ("2003-03-21", "Iraq War"),
        ("2008-09-19", "Lehman Brothers"),
        ("2020-03-20", "COVID crash"),
        ("2022-03-01", "Russia/Ukraine"),
        ("2023-03-15", "SVB crisis"),
    ]
    for date_str, label in checks:
        active = get_active_events(date_str)
        events_str = ", ".join(f"{t}({i:.1f})" for t, i, _ in active)
        print(f"    {date_str} [{label:20s}]: {events_str}")


if __name__ == '__main__':
    print_summary()
