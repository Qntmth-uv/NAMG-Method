using LaTeXStrings
using Plots; pythonplot()
using LinearAlgebra

#Change  the plotting function to GR instead of matplotlib. There is a problem with matplotlib in the creation of the 
#levels graph.
gr()

function plot_optimization_path(f::Function, eval_points::AbstractMatrix; function_name::String = "Function ", padding_ratio::Float64 = 0.2, levels::Int = 30, 
                                resolution::Int = 100, label::String = nothing, g_xlims, g_ylims)

    # 1. Definimos lógica de expansión en una mini-función (Calcula límites nuevos + grid)
    setup_axis(l) = let d = max(l[2]-l[1], 1.0) * padding_ratio, new_l = (l[1]-d, l[2]+d)
        new_l, range(new_l..., length=resolution)
    end

    (xlims, x_grid), (ylims, y_grid) = setup_axis.([g_xlims, g_ylims])

    Z = [f([x, y]) for y in y_grid, x in x_grid]

    min_z, max_z = extrema(Z)
    if min_z < 0
        lvl_vals = range(min_z, max_z, length=levels)
    else
        start_val = max(min_z, 1e-5)
        lvl_vals = 10 .^ range(log10(start_val), log10(max_z), length=levels)
    end

    #Create the graph levles in the given space
    p = contour(x_grid, y_grid, Z, 
        levels=lvl_vals, color=:plasma, fill=false, alpha=0.4, 
        clabels=false, cbar=false, label=nothing,
        title="$(function_name) Landscape", xlabel="x", ylabel="y")

    add_optimization_path!(p, eval_points, label=label, color=:crimson)
    
    return p
end


function add_optimization_path!(plt::Plots.Plot, eval_points::AbstractMatrix; label::String="Path", color::Symbol=:crimson, linestyle::Symbol = :solid, markersize::Int = 5, 
                                alpha::Float16=0.7, init_end_points_alpha::Float16 = 1.0) 
    # Extraemos coordenadas (Igual que en tu código original)
    xs_path = eval_points[:, 1]
    ys_path = eval_points[:, 2]

    if !isempty(eval_points)
        
        #Path construction
        plot!(plt, xs_path, ys_path, label=label, linecolor=color, linestyle = linestyle, linewidth=1.5, marker=:circle, markercolor=color,markersize=1, alpha=alpha)

        #Initial point
        scatter!(plt, [xs_path[1]], [ys_path[1]], markercolor=:green, markerstrokecolor=:black, marker=:circle, markersize=markersize, label=nothing, alpha=init_end_points_alpha)
        
        #End point
        scatter!(plt, [xs_path[end]], [ys_path[end]], markercolor=color, markerstrokecolor=:black, marker=:star5, markersize=markersize5, label=nothing, alpha=init_end_points_alpha)
    end
    return plt
end

function plot_area_optimization_paths(f::Function, array_paths::AbstractArray, label::String, color::Symbol; name_f::String = "Function", padding_ratio::Float64 = 0.2,
                                    number_of_levels::Int = 30, canvas_limitX, canvas_limitY, image_path::String = "path_area_optimization.svg")

    config_canvas = Dict(function_name => name_f, padding_ratio => padding_ratio, levels => number_of_levels, label => label, g_xlims => canvas_limitX, g_ylims => canvas_limitY)
    
    canvas = plot_optimization_path(f, array_paths[1]; config_canvas)
    number_paths = length(array_paths)
    for i in (1:number_paths)
        add_optimization_path!(canvas, array_paths[i], "", color, :solid, markersize, 0.1, 0.1)
    end
    #Save the plot
    savefig(canvas, image_path)
end


function plotEvolution(list_of_Historials, colors, labels, styleLine, ylabel::String, plotTitle::String)
    """
    # Definition
    Function to plot the evolution of a phenomena. 
    
    ## Inputs

    - list_of_Historials: Array[Vectors] - Array of several evolutions of the phenomena
    - colors: Array - Array of colors for each historial (e. g. :blue, :red)
    - labels: Array - Array of labels for each historial 
    - styline: Array - Array of lineStyles to plot the historial (e. g; :dash, :solid)
    - ylabel: String - Name of the Y axis
    - plotTitle: String - Title of the plot
    - image_name: String - Path where to save the plot

    ## Output:

    - None: Void - Only creates the plot and save it
        
    ## Remark 
    There is not a check if the length of the arrays of evolution is the same as for the colors
    and the styleLine. If the lengths does not match, then the function will raise a Error.
    """
    #Number of the elements in the historial 
    n_elemnts = length(list_of_Historials)

    #Configuration of the axis of the plot
    plt = plot(
        xlabel="Iteration",
        ylabel=ylabel,
        title=plotTitle,
        yscale=:log10,
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3,
        legend=:topright,
        size=(800, 400),
        dpi=300
    )

    #Add each of the lines
    for i in 1:n_elemnts
        plot!(plt, list_of_Historials[i],
            label = labels[i],
            linestyle = styleLine[i],
            color = colors[i],
            linewidth = 1.5
        )
    end
    return plt
end