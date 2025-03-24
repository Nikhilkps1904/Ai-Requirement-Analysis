import pandas as pd
import os

# Define the 20 conflict types
conflict_types = [
    "Performance Conflict", "Compliance Conflict", "Safety Conflict", "Cost Conflict", "Battery Conflict",
    "Environmental Conflict", "Structural Conflict", "Comfort Conflict", "Power Source Conflict", "Reliability Conflict",
    "Scalability Conflict", "Security Conflict", "Usability Conflict", "Maintenance Conflict", "Weight Conflict",
    "Time-to-Market Conflict", "Compatibility Conflict", "Aesthetic Conflict", "Noise Conflict", "Other Conflict"
]

# Predefined requirement templates for each conflict type (10 examples per type)
data = [
    # Performance Conflict (10)
    {"Requirement_1": "The bike must reach a top speed of 180 km/h.", "Requirement_2": "The vehicle must achieve 50 km/l fuel efficiency.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The vehicle must accelerate to 100 km/h in 5 seconds.", "Requirement_2": "The engine must prioritize low fuel consumption.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The bike must have a high-performance turbo engine.", "Requirement_2": "The vehicle must maintain 40 mpg efficiency.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The vehicle must reach 200 km/h top speed.", "Requirement_2": "The bike must use a lightweight fuel-efficient motor.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The bike must deliver 35 HP output.", "Requirement_2": "The vehicle must achieve 60 km/l efficiency.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The vehicle must have a sport-tuned suspension.", "Requirement_2": "The bike must optimize for fuel savings.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The bike must reach 150 km/h.", "Requirement_2": "The vehicle must use a low-power eco engine.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The vehicle must include a high-RPM engine.", "Requirement_2": "The bike must achieve 45 km/l efficiency.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The bike must have rapid acceleration.", "Requirement_2": "The vehicle must prioritize energy efficiency.", "Conflict_Type": "Performance Conflict"},
    {"Requirement_1": "The vehicle must reach 120 km/h in 6 seconds.", "Requirement_2": "The bike must maintain high fuel economy.", "Conflict_Type": "Performance Conflict"},

    # Compliance Conflict (10)
    {"Requirement_1": "The vehicle must meet strict emission standards.", "Requirement_2": "The bike must have a high-output engine.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The bike must comply with urban noise limits.", "Requirement_2": "The vehicle must feature a loud exhaust system.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The vehicle must adhere to safety regulations.", "Requirement_2": "The bike must allow manual headlight control.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The bike must meet EU crash test standards.", "Requirement_2": "The vehicle must use lightweight materials.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The vehicle must follow emissions Tier 4.", "Requirement_2": "The bike must prioritize performance tuning.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The bike must have always-on daytime lights.", "Requirement_2": "The vehicle must allow light customization.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The vehicle must comply with noise restrictions.", "Requirement_2": "The bike must have a sporty sound profile.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The bike must meet regional safety codes.", "Requirement_2": "The vehicle must use minimal safety features.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The vehicle must follow EPA standards.", "Requirement_2": "The bike must use a high-power gas engine.", "Conflict_Type": "Compliance Conflict"},
    {"Requirement_1": "The bike must adhere to lighting laws.", "Requirement_2": "The vehicle must offer switchable lights.", "Conflict_Type": "Compliance Conflict"},

    # Safety Conflict (10)
    {"Requirement_1": "The vehicle must include side-impact airbags.", "Requirement_2": "The bike must weigh under 130 kg.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The bike must have ABS brakes.", "Requirement_2": "The vehicle must be ultra-lightweight.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The vehicle must feature a roll cage.", "Requirement_2": "The bike must maintain a slim profile.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The bike must include a crash sensor system.", "Requirement_2": "The vehicle must minimize weight.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The vehicle must have reinforced bumpers.", "Requirement_2": "The bike must be compact.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The bike must use a sturdy steel frame.", "Requirement_2": "The vehicle must weigh under 140 kg.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The vehicle must include emergency braking.", "Requirement_2": "The bike must prioritize low weight.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The bike must have a helmet lock system.", "Requirement_2": "The vehicle must avoid added components.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The vehicle must feature a stability control.", "Requirement_2": "The bike must be lightweight.", "Conflict_Type": "Safety Conflict"},
    {"Requirement_1": "The bike must include a tire pressure monitor.", "Requirement_2": "The vehicle must reduce mass.", "Conflict_Type": "Safety Conflict"},

    # Cost Conflict (10)
    {"Requirement_1": "The bike must cost under $1,200.", "Requirement_2": "The vehicle must include a digital dashboard.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The vehicle must use premium alloy wheels.", "Requirement_2": "The bike must stay under $2,000.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The bike must have a luxury paint finish.", "Requirement_2": "The vehicle must be budget-friendly.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The vehicle must include adaptive headlights.", "Requirement_2": "The bike must cost below $1,500.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The bike must use high-end shock absorbers.", "Requirement_2": "The vehicle must be low-cost.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The vehicle must have a custom exhaust.", "Requirement_2": "The bike must stay under $1,800.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The bike must include a leather seat.", "Requirement_2": "The vehicle must be affordable.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The vehicle must feature a premium sound system.", "Requirement_2": "The bike must cost under $2,500.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The bike must have a titanium frame.", "Requirement_2": "The vehicle must minimize production costs.", "Conflict_Type": "Cost Conflict"},
    {"Requirement_1": "The vehicle must include a navigation system.", "Requirement_2": "The bike must be priced under $1,000.", "Conflict_Type": "Cost Conflict"},

    # Battery Conflict (10)
    {"Requirement_1": "The vehicle must have a 350 km electric range.", "Requirement_2": "The bike must use a small battery pack.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The bike must charge to 80% in 20 minutes.", "Requirement_2": "The vehicle must use a lightweight battery.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The vehicle must include a high-capacity battery.", "Requirement_2": "The bike must fit a compact design.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The bike must offer 400 km range.", "Requirement_2": "The vehicle must minimize battery size.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The vehicle must support fast charging.", "Requirement_2": "The bike must use a slim battery.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The bike must have a 300 km range.", "Requirement_2": "The vehicle must reduce battery weight.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The vehicle must use a long-life battery.", "Requirement_2": "The bike must have a small footprint.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The bike must include a swappable battery.", "Requirement_2": "The vehicle must limit battery space.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The vehicle must offer 250 km range.", "Requirement_2": "The bike must use a minimal battery.", "Conflict_Type": "Battery Conflict"},
    {"Requirement_1": "The bike must charge in 15 minutes.", "Requirement_2": "The vehicle must have a compact battery.", "Conflict_Type": "Battery Conflict"},

    # Environmental Conflict (10)
    {"Requirement_1": "The vehicle must use eco-friendly materials.", "Requirement_2": "The bike must have a 40 HP engine.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The bike must minimize carbon emissions.", "Requirement_2": "The vehicle must use a high-performance motor.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The vehicle must be fully recyclable.", "Requirement_2": "The bike must include a powerful gas engine.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The bike must meet green certification.", "Requirement_2": "The vehicle must prioritize speed.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The vehicle must reduce emissions.", "Requirement_2": "The bike must have a turbocharged engine.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The bike must use biodegradable plastics.", "Requirement_2": "The vehicle must deliver 35 HP.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The vehicle must follow eco standards.", "Requirement_2": "The bike must use a loud exhaust.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The bike must have zero-emission tech.", "Requirement_2": "The vehicle must prioritize power output.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The vehicle must use sustainable fuel.", "Requirement_2": "The bike must reach 180 km/h.", "Conflict_Type": "Environmental Conflict"},
    {"Requirement_1": "The bike must minimize environmental impact.", "Requirement_2": "The vehicle must have a 30 HP engine.", "Conflict_Type": "Environmental Conflict"},

    # Structural Conflict (10)
    {"Requirement_1": "The vehicle must use a rigid aluminum frame.", "Requirement_2": "The bike must weigh under 120 kg.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The bike must have a reinforced chassis.", "Requirement_2": "The vehicle must be lightweight.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The vehicle must include a steel roll bar.", "Requirement_2": "The bike must stay under 110 kg.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The bike must use a durable composite body.", "Requirement_2": "The vehicle must reduce weight.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The vehicle must have a solid frame design.", "Requirement_2": "The bike must weigh under 130 kg.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The bike must include a heavy-duty chassis.", "Requirement_2": "The vehicle must be light.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The vehicle must use a thick steel structure.", "Requirement_2": "The bike must minimize weight.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The bike must have a robust frame.", "Requirement_2": "The vehicle must weigh under 140 kg.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The vehicle must include a reinforced body.", "Requirement_2": "The bike must be ultra-light.", "Conflict_Type": "Structural Conflict"},
    {"Requirement_1": "The bike must use a strong alloy frame.", "Requirement_2": "The vehicle must stay under 150 kg.", "Conflict_Type": "Structural Conflict"},

    # Comfort Conflict (10)
    {"Requirement_1": "The vehicle must have adjustable cushioned seats.", "Requirement_2": "The bike must minimize power usage.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The bike must include a climate-controlled cabin.", "Requirement_2": "The vehicle must reduce energy draw.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The vehicle must offer a plush interior.", "Requirement_2": "The bike must use a low-power system.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The bike must have ergonomic handlebars.", "Requirement_2": "The vehicle must avoid extra power needs.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The vehicle must include a heated steering wheel.", "Requirement_2": "The bike must conserve battery.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The bike must offer a soft suspension.", "Requirement_2": "The vehicle must limit energy use.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The vehicle must have luxury upholstery.", "Requirement_2": "The bike must reduce power consumption.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The bike must include padded grips.", "Requirement_2": "The vehicle must optimize efficiency.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The vehicle must offer a quiet ride.", "Requirement_2": "The bike must minimize electrical load.", "Conflict_Type": "Comfort Conflict"},
    {"Requirement_1": "The bike must have a reclining seat.", "Requirement_2": "The vehicle must save energy.", "Conflict_Type": "Comfort Conflict"},

    # Power Source Conflict (10)
    {"Requirement_1": "The vehicle must use a diesel engine.", "Requirement_2": "The bike must have a 200 km electric range.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The bike must be fully electric.", "Requirement_2": "The vehicle must use a gasoline motor.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The vehicle must run on hybrid power.", "Requirement_2": "The bike must be purely gas-powered.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The bike must use a fuel cell system.", "Requirement_2": "The vehicle must rely on a petrol engine.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The vehicle must have an electric motor.", "Requirement_2": "The bike must use diesel fuel.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The bike must run on battery power.", "Requirement_2": "The vehicle must use a combustion engine.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The vehicle must be hydrogen-powered.", "Requirement_2": "The bike must use gasoline.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The bike must have a plug-in electric system.", "Requirement_2": "The vehicle must use a diesel motor.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The vehicle must use a dual-fuel system.", "Requirement_2": "The bike must be electric-only.", "Conflict_Type": "Power Source Conflict"},
    {"Requirement_1": "The bike must rely on solar charging.", "Requirement_2": "The vehicle must use a gas engine.", "Conflict_Type": "Power Source Conflict"},

    # Reliability Conflict (10)
    {"Requirement_1": "The vehicle must offer a 7-year warranty.", "Requirement_2": "The bike must use cheap plastic parts.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The bike must last 15 years.", "Requirement_2": "The vehicle must use low-durability materials.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The vehicle must have a robust engine.", "Requirement_2": "The bike must use disposable components.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The bike must ensure 100,000 km lifespan.", "Requirement_2": "The vehicle must cut material costs.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The vehicle must be highly durable.", "Requirement_2": "The bike must use lightweight, weak parts.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The bike must have a reliable battery.", "Requirement_2": "The vehicle must use low-cost cells.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The vehicle must withstand harsh conditions.", "Requirement_2": "The bike must use budget materials.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The bike must offer long-term reliability.", "Requirement_2": "The vehicle must reduce part quality.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The vehicle must have a 5-year guarantee.", "Requirement_2": "The bike must use short-life plastics.", "Conflict_Type": "Reliability Conflict"},
    {"Requirement_1": "The bike must ensure engine longevity.", "Requirement_2": "The vehicle must use low-grade steel.", "Conflict_Type": "Reliability Conflict"},

    # Scalability Conflict (10)
    {"Requirement_1": "The vehicle must support multiple configurations.", "Requirement_2": "The bike must weigh under 110 kg.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The bike must allow modular upgrades.", "Requirement_2": "The vehicle must be ultra-light.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The vehicle must offer expandable features.", "Requirement_2": "The bike must minimize weight.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The bike must support add-on accessories.", "Requirement_2": "The vehicle must stay under 120 kg.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The vehicle must have a scalable platform.", "Requirement_2": "The bike must reduce mass.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The bike must allow future expansions.", "Requirement_2": "The vehicle must be lightweight.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The vehicle must support variant designs.", "Requirement_2": "The bike must weigh under 130 kg.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The bike must have a flexible chassis.", "Requirement_2": "The vehicle must limit weight.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The vehicle must offer upgrade options.", "Requirement_2": "The bike must stay under 140 kg.", "Conflict_Type": "Scalability Conflict"},
    {"Requirement_1": "The bike must support scalability.", "Requirement_2": "The vehicle must be compact.", "Conflict_Type": "Scalability Conflict"},

    # Security Conflict (10)
    {"Requirement_1": "The vehicle must have a GPS tracking system.", "Requirement_2": "The bike must use a basic lock.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The bike must include an alarm system.", "Requirement_2": "The vehicle must avoid electronics.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The vehicle must use a smart key.", "Requirement_2": "The bike must have a manual ignition.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The bike must feature remote locking.", "Requirement_2": "The vehicle must use a simple key.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The vehicle must have an anti-theft device.", "Requirement_2": "The bike must minimize tech.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The bike must use a fingerprint scanner.", "Requirement_2": "The vehicle must be low-tech.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The vehicle must include a security camera.", "Requirement_2": "The bike must avoid complexity.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The bike must have a theft alert system.", "Requirement_2": "The vehicle must use basic security.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The vehicle must feature a coded lock.", "Requirement_2": "The bike must use a standard key.", "Conflict_Type": "Security Conflict"},
    {"Requirement_1": "The bike must include a tracking chip.", "Requirement_2": "The vehicle must be simple.", "Conflict_Type": "Security Conflict"},

    # Usability Conflict (10)
    {"Requirement_1": "The vehicle must have an intuitive voice control.", "Requirement_2": "The bike must use manual controls only.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The bike must feature a digital interface.", "Requirement_2": "The vehicle must be mechanically simple.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The vehicle must offer a heads-up display.", "Requirement_2": "The bike must avoid screens.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The bike must have automated gear shifting.", "Requirement_2": "The vehicle must use a manual system.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The vehicle must include a touch panel.", "Requirement_2": "The bike must prioritize simplicity.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The bike must offer a smart dashboard.", "Requirement_2": "The vehicle must use analog gauges.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The vehicle must have a user-friendly app.", "Requirement_2": "The bike must avoid connectivity.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The bike must include a navigation screen.", "Requirement_2": "The vehicle must be basic.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The vehicle must feature auto-adjust seats.", "Requirement_2": "The bike must use fixed seating.", "Conflict_Type": "Usability Conflict"},
    {"Requirement_1": "The bike must have a customizable UI.", "Requirement_2": "The vehicle must stay simple.", "Conflict_Type": "Usability Conflict"},

    # Maintenance Conflict (10)
    {"Requirement_1": "The vehicle must need service every 25,000 km.", "Requirement_2": "The bike must use a high-maintenance engine.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The bike must have a self-repairing tire system.", "Requirement_2": "The vehicle must require frequent checks.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The vehicle must last 5 years without service.", "Requirement_2": "The bike must use a complex motor.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The bike must need minimal upkeep.", "Requirement_2": "The vehicle must include intricate parts.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The vehicle must have a low-maintenance design.", "Requirement_2": "The bike must use a hybrid system.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The bike must require service every 30,000 km.", "Requirement_2": "The vehicle must have a delicate engine.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The vehicle must avoid regular maintenance.", "Requirement_2": "The bike must use high-wear components.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The bike must have a durable finish.", "Requirement_2": "The vehicle must need constant care.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The vehicle must use long-life parts.", "Requirement_2": "The bike must require frequent repairs.", "Conflict_Type": "Maintenance Conflict"},
    {"Requirement_1": "The bike must minimize service needs.", "Requirement_2": "The vehicle must use fragile tech.", "Conflict_Type": "Maintenance Conflict"},

    # Weight Conflict (10)
    {"Requirement_1": "The vehicle must weigh under 110 kg.", "Requirement_2": "The bike must include a heavy battery.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The bike must stay under 120 kg.", "Requirement_2": "The vehicle must use a steel frame.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The vehicle must be ultra-light at 100 kg.", "Requirement_2": "The bike must have a large fuel tank.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The bike must weigh under 130 kg.", "Requirement_2": "The vehicle must include a robust chassis.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The vehicle must minimize weight to 140 kg.", "Requirement_2": "The bike must use a heavy motor.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The bike must be under 150 kg.", "Requirement_2": "The vehicle must have a reinforced body.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The vehicle must weigh under 115 kg.", "Requirement_2": "The bike must include extra storage.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The bike must stay light at 125 kg.", "Requirement_2": "The vehicle must use a solid frame.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The vehicle must be under 135 kg.", "Requirement_2": "The bike must have a big battery.", "Conflict_Type": "Weight Conflict"},
    {"Requirement_1": "The bike must weigh under 145 kg.", "Requirement_2": "The vehicle must include a heavy suspension.", "Conflict_Type": "Weight Conflict"},

    # Time-to-Market Conflict (10)
    {"Requirement_1": "The vehicle must launch in 4 months.", "Requirement_2": "The bike must pass rigorous testing.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The bike must be ready in 6 months.", "Requirement_2": "The vehicle must meet new safety laws.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The vehicle must hit the market in 8 months.", "Requirement_2": "The bike must undergo long trials.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The bike must launch in 5 months.", "Requirement_2": "The vehicle must certify emissions.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The vehicle must be out in 7 months.", "Requirement_2": "The bike must complete durability tests.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The bike must launch in 9 months.", "Requirement_2": "The vehicle must pass crash tests.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The vehicle must be ready in 3 months.", "Requirement_2": "The bike must meet strict standards.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The bike must hit stores in 10 months.", "Requirement_2": "The vehicle must finish R&D.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The vehicle must launch in 12 months.", "Requirement_2": "The bike must pass global regs.", "Conflict_Type": "Time-to-Market Conflict"},
    {"Requirement_1": "The bike must be out in 11 months.", "Requirement_2": "The vehicle must test extensively.", "Conflict_Type": "Time-to-Market Conflict"},

    # Compatibility Conflict (10)
    {"Requirement_1": "The vehicle must use USB-C charging.", "Requirement_2": "The bike must support legacy ports.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The bike must integrate with Android Auto.", "Requirement_2": "The vehicle must use iOS only.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The vehicle must support universal tires.", "Requirement_2": "The bike must use custom wheels.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The bike must use a standard battery.", "Requirement_2": "The vehicle must have a unique pack.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The vehicle must work with old accessories.", "Requirement_2": "The bike must use new tech.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The bike must support Bluetooth 5.0.", "Requirement_2": "The vehicle must use Bluetooth 4.0.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The vehicle must use a common charger.", "Requirement_2": "The bike must have a proprietary plug.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The bike must fit standard racks.", "Requirement_2": "The vehicle must use custom mounts.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The vehicle must support open-source software.", "Requirement_2": "The bike must use closed systems.", "Conflict_Type": "Compatibility Conflict"},
    {"Requirement_1": "The bike must use universal connectors.", "Requirement_2": "The vehicle must have unique ports.", "Conflict_Type": "Compatibility Conflict"},

    # Aesthetic Conflict (10)
    {"Requirement_1": "The vehicle must have a sleek modern look.", "Requirement_2": "The bike must use a boxy frame.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The bike must feature a vintage style.", "Requirement_2": "The vehicle must be aerodynamic.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The vehicle must use sharp angular lines.", "Requirement_2": "The bike must have a rounded design.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The bike must have a minimalist look.", "Requirement_2": "The vehicle must include ornate details.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The vehicle must feature a retro finish.", "Requirement_2": "The bike must optimize airflow.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The bike must use a bold color scheme.", "Requirement_2": "The vehicle must be subtle.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The vehicle must have a futuristic design.", "Requirement_2": "The bike must look classic.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The bike must feature chrome accents.", "Requirement_2": "The vehicle must be streamlined.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The vehicle must use a matte finish.", "Requirement_2": "The bike must have glossy paint.", "Conflict_Type": "Aesthetic Conflict"},
    {"Requirement_1": "The bike must have a rugged aesthetic.", "Requirement_2": "The vehicle must be sleek.", "Conflict_Type": "Aesthetic Conflict"},

    # Noise Conflict (10)
    {"Requirement_1": "The vehicle must have a quiet electric motor.", "Requirement_2": "The bike must produce a loud roar.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The bike must meet silent operation standards.", "Requirement_2": "The vehicle must have a sport exhaust.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The vehicle must reduce noise to 50 dB.", "Requirement_2": "The bike must sound aggressive.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The bike must operate silently.", "Requirement_2": "The vehicle must use a noisy engine.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The vehicle must comply with 60 dB limits.", "Requirement_2": "The bike must have a bold sound.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The bike must be noise-free.", "Requirement_2": "The vehicle must feature a loud muffler.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The vehicle must minimize sound output.", "Requirement_2": "The bike must enhance exhaust noise.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The bike must stay under 55 dB.", "Requirement_2": "The vehicle must sound powerful.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The vehicle must use a silent system.", "Requirement_2": "The bike must have a deep tone.", "Conflict_Type": "Noise Conflict"},
    {"Requirement_1": "The bike must meet quiet zone rules.", "Requirement_2": "The vehicle must amplify sound.", "Conflict_Type": "Noise Conflict"},

    # Other Conflict (10)
    {"Requirement_1": "The vehicle must include a unique logo.", "Requirement_2": "The bike must reduce branding costs.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The bike must offer custom decals.", "Requirement_2": "The vehicle must keep production simple.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The vehicle must have a special edition trim.", "Requirement_2": "The bike must stay low-cost.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The bike must include a rare feature.", "Requirement_2": "The vehicle must avoid extras.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The vehicle must use a distinct color.", "Requirement_2": "The bike must use standard paint.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The bike must have a limited-run design.", "Requirement_2": "The vehicle must be mass-produced.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The vehicle must offer a niche accessory.", "Requirement_2": "The bike must cut costs.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The bike must include a bespoke part.", "Requirement_2": "The vehicle must simplify assembly.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The vehicle must have a custom interior.", "Requirement_2": "The bike must reduce expenses.", "Conflict_Type": "Other Conflict"},
    {"Requirement_1": "The bike must feature a signature look.", "Requirement_2": "The vehicle must be generic.", "Conflict_Type": "Other Conflict"}
]

# Create a DataFrame
df = pd.DataFrame(data, columns=["Requirement_1", "Requirement_2", "Conflict_Type"])

# Ensure the output directory exists
output_dir = "Results/CSV"
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
csv_path = os.path.join(output_dir, "optimized_training_requirements.csv")
df.to_csv(csv_path, index=False)

print(f"Training CSV file saved to {csv_path}")
print(f"Number of rows: {len(df)}")
print(f"Unique conflict types: {len(df['Conflict_Type'].unique())}")
print("Conflict type distribution:")
print(df['Conflict_Type'].value_counts())