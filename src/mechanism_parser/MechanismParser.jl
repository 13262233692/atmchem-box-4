module MechanismParser

export Species, Reaction, ChemicalMechanism, parse_mechanism, parse_kpp_mechanism

struct Species
    name::String
    formula::String
    index::Int
    is_constant::Bool
end

struct RateCoefficient
    type::String
    parameters::Vector{Float64}
end

struct Reaction
    id::String
    reactants::Vector{Tuple{String, Int}}
    products::Vector{Tuple{String, Int}}
    rate::RateCoefficient
    is_photolysis::Bool
    photolysis_label::Union{String, Nothing}
end

struct ChemicalMechanism
    name::String
    species::Vector{Species}
    reactions::Vector{Reaction}
    species_index::Dict{String, Int}
end

function ChemicalMechanism(name::String, species::Vector{Species}, reactions::Vector{Reaction})
    species_index = Dict(s.name => s.index for s in species)
    return ChemicalMechanism(name, species, reactions, species_index)
end

function parse_mechanism(filepath::String)
    ext = lowercase(splitext(filepath)[2])
    if ext == ".kpp"
        return parse_kpp_mechanism(filepath)
    elseif ext == ".yaml" || ext == ".yml"
        return parse_yaml_mechanism(filepath)
    else
        error("Unsupported mechanism file format: $ext")
    end
end

function parse_kpp_mechanism(filepath::String)
    content = read(filepath, String)
    lines = split(content, '\n')
    
    species = Species[]
    reactions = Reaction[]
    species_idx = 1
    reaction_id = 1
    
    in_defs = false
    in_eqns = false
    
    for line in lines
        line = strip(line)
        isempty(line) && continue
        startswith(line, '!') && continue
        
        if occursin(r"^#DEFINES", line)
            in_defs = true
            in_eqns = false
            continue
        elseif occursin(r"^#EQUATIONS", line)
            in_defs = false
            in_eqns = true
            continue
        elseif startswith(line, '#')
            in_defs = false
            in_eqns = false
            continue
        end
        
        if in_defs
            parts = split(line, r"\s+")
            for part in parts
                if !isempty(part) && !startswith(part, '!')
                    is_const = occursin(r"\{.*constant.*\}", part)
                    name = replace(part, r"\{.*\}" => "")
                    name = strip(name, [';', ' '])
                    if !isempty(name)
                        push!(species, Species(name, name, species_idx, is_const))
                        species_idx += 1
                    end
                end
            end
        elseif in_eqns
            if occursin('=', line)
                eqn_parts = split(line, ':')
                rate_part = strip(eqn_parts[1])
                eqn_part = strip(join(eqn_parts[2:end], ':'))
                eqn_part = replace(eqn_part, ';' => "")
                
                is_photo = startswith(rate_part, "J")
                photo_label = is_photo ? rate_part : nothing
                
                rate = is_photo ? 
                    RateCoefficient("photolysis", [0.0]) :
                    parse_rate_expression(rate_part)
                
                reactants, products = parse_reaction_equation(eqn_part)
                
                push!(reactions, Reaction(
                    "R$(reaction_id)",
                    reactants,
                    products,
                    rate,
                    is_photo,
                    photo_label
                ))
                reaction_id += 1
            end
        end
    end
    
    return ChemicalMechanism(basename(filepath), species, reactions)
end

function parse_rate_expression(expr::String)
    expr = strip(expr)
    
    if occursin(r"^ARR", expr)
        m = match(r"ARR\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)", expr)
        if m !== nothing
            A = parse(Float64, strip(m.captures[1]))
            B = parse(Float64, strip(m.captures[2]))
            E = parse(Float64, strip(m.captures[3]))
            return RateCoefficient("arrhenius", [A, B, E])
        end
    end
    
    if occursin(r"^EXP", expr)
        m = match(r"EXP\s*\(\s*([^)]+)\)", expr)
        if m !== nothing
            val = parse(Float64, strip(m.captures[1]))
            return RateCoefficient("exp", [val])
        end
    end
    
    m = match(r"([\d\.eE\+\-]+)", expr)
    if m !== nothing
        k = parse(Float64, m.captures[1])
        return RateCoefficient("constant", [k])
    end
    
    return RateCoefficient("constant", [1e-10])
end

function parse_reaction_equation(eqn::String)
    eqn = replace(eqn, r"\s+" => "")
    
    if occursin('=', eqn)
        lhs, rhs = split(eqn, '=')
    elseif occursin("->", eqn)
        lhs, rhs = split(eqn, "->")
    elseif occursin("→", eqn)
        lhs, rhs = split(eqn, "→")
    else
        return [], []
    end
    
    reactants = parse_species_list(lhs)
    products = parse_species_list(rhs)
    
    return reactants, products
end

function parse_species_list(str::String)
    species_list = Tuple{String, Int}[]
    parts = split(str, '+')
    
    for part in parts
        part = strip(part)
        isempty(part) && continue
        
        m = match(r"^(\d*\.?\d*)\s*([A-Za-z_]\w*)", part)
        if m !== nothing
            coeff_str = m.captures[1]
            name = m.captures[2]
            coeff = isempty(coeff_str) ? 1 : parse(Int, coeff_str)
            push!(species_list, (name, coeff))
        elseif occursin(r"^[A-Za-z_]", part)
            push!(species_list, (part, 1))
        end
    end
    
    return species_list
end

function parse_yaml_mechanism(filepath::String)
    using YAML
    data = YAML.load_file(filepath)
    
    species = Species[]
    reactions = Reaction[]
    
    for (i, spec_data) in enumerate(data["species"])
        name = spec_data["name"]
        formula = get(spec_data, "formula", name)
        is_const = get(spec_data, "constant", false)
        push!(species, Species(name, formula, i, is_const))
    end
    
    for (i, rxn_data) in enumerate(data["reactions"])
        reactants = [(r["species"], get(r, "coefficient", 1)) for r in rxn_data["reactants"]]
        products = [(p["species"], get(p, "coefficient", 1)) for p in rxn_data["products"]]
        
        rate_data = rxn_data["rate"]
        rate_type = rate_data["type"]
        rate_params = Float64.(rate_data["parameters"])
        rate = RateCoefficient(rate_type, rate_params)
        
        is_photo = get(rxn_data, "photolysis", false)
        photo_label = get(rxn_data, "photolysis_label", nothing)
        
        push!(reactions, Reaction(
            get(rxn_data, "id", "R$i"),
            reactants,
            products,
            rate,
            is_photo,
            photo_label
        ))
    end
    
    return ChemicalMechanism(data["name"], species, reactions)
end

end 
