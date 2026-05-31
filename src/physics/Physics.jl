module Physics

import ..MechanismParser: ChemicalMechanism

export Photolysis, Emissions, Deposition, 
       update_photolysis!, set_photolysis_rates!,
       set_emissions!, set_deposition_rates!

const SECONDS_PER_DAY = 86400.0

struct Photolysis
    n_reactions::Int
    rates::Vector{Float64}
    labels::Vector{String}
    diurnal_profile::Bool
    solar_noon::Float64
    latitude::Float64
end

function Photolysis(mechanism::ChemicalMechanism)
    n_photo = sum(r.is_photolysis for r in mechanism.reactions)
    labels = [r.photolysis_label for r in mechanism.reactions if r.is_photolysis]
    return Photolysis(
        n_photo,
        zeros(n_photo),
        labels,
        true,
        12.0,
        45.0
    )
end

function update_photolysis!(photo::Photolysis, t::Float64, T::Float64=298.15)
    if photo.diurnal_profile
        hour_of_day = (t / 3600.0) % 24.0
        zenith = compute_solar_zenith_angle(hour_of_day, photo.solar_noon, photo.latitude)
        scale_factor = max(0.0, cos(zenith * pi / 180.0))^0.5
        for i in 1:photo.n_reactions
            photo.rates[i] *= scale_factor
        end
    end
    return photo.rates
end

function compute_solar_zenith_angle(hour::Float64, solar_noon::Float64, latitude::Float64)
    hour_angle = 15.0 * (hour - solar_noon)
    declination = 0.0
    
    sin_zenith = sin(latitude * pi / 180.0) * sin(declination * pi / 180.0) +
                 cos(latitude * pi / 180.0) * cos(declination * pi / 180.0) *
                 cos(hour_angle * pi / 180.0)
    sin_zenith = clamp(sin_zenith, -1.0, 1.0)
    
    zenith = acos(sin_zenith) * 180.0 / pi
    return zenith
end

function set_photolysis_rates!(photo::Photolysis, rate_dict::Dict)
    for (label, rate) in rate_dict
        idx = findfirst(==(label), photo.labels)
        if idx !== nothing
            photo.rates[idx] = rate
        end
    end
end

struct Emissions
    n_species::Int
    rates::Vector{Float64}
    species_names::Vector{String}
    diurnal_profile::Bool
    scale_factors::Vector{Float64}
end

function Emissions(mechanism::ChemicalMechanism)
    n_spec = length(mechanism.species)
    names = [s.name for s in mechanism.species]
    return Emissions(
        n_spec,
        zeros(n_spec),
        names,
        false,
        ones(n_spec)
    )
end

function set_emissions!(emis::Emissions, emission_dict::Dict)
    for (name, rate) in emission_dict
        idx = findfirst(==(name), emis.species_names)
        if idx !== nothing
            emis.rates[idx] = rate
        end
    end
end

function update_emissions!(emis::Emissions, t::Float64)
    if emis.diurnal_profile
        hour_of_day = (t / 3600.0) % 24.0
        for i in 1:emis.n_species
            if 6.0 <= hour_of_day <= 22.0
                emis.scale_factors[i] = 1.0
            else
                emis.scale_factors[i] = 0.1
            end
        end
    end
    return emis.rates .* emis.scale_factors
end

struct Deposition
    n_species::Int
    rates::Vector{Float64}
    species_names::Vector{String}
    deposition_velocities::Vector{Float64}
    height::Float64
end

function Deposition(mechanism::ChemicalMechanism; height::Float64=1000.0)
    n_spec = length(mechanism.species)
    names = [s.name for s in mechanism.species]
    return Deposition(
        n_spec,
        zeros(n_spec),
        names,
        zeros(n_spec),
        height
    )
end

function set_deposition_rates!(dep::Deposition, dep_dict::Dict)
    for (name, velocity) in dep_dict
        idx = findfirst(==(name), dep.species_names)
        if idx !== nothing
            dep.deposition_velocities[idx] = velocity
            dep.rates[idx] = velocity / dep.height
        end
    end
end

function set_deposition_velocity!(dep::Deposition, velocity_dict::Dict)
    for (name, velocity) in velocity_dict
        idx = findfirst(==(name), dep.species_names)
        if idx !== nothing
            dep.deposition_velocities[idx] = velocity
            dep.rates[idx] = velocity / dep.height
        end
    end
end

end 
