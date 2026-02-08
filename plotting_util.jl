using LaTeXStrings
using Plots; pythonplot()
using LinearAlgebra

#Change  the ploting function to GR instead of matplotlib. There is a problem with matplotlib in the creation of the 
#levels graph.
gr()

function plot_optimization_path(f::Function, eval_points::AbstractMatrix; 
                                function_name::String = "Function ", 
                                padding_ratio::Float64 = 0.2, 
                                levels::Int = 30, 
                                resolution::Int = 100,
                                label = nothing,
                                g_xlims, g_ylims)

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

"""

add_optimization_path!(plt, eval_points; ...)
Añade un nuevo camino de optimización a un gráfico existente 'plt'.

"""
function add_optimization_path!(plt::Plots.Plot, eval_points::AbstractMatrix; 
                                label::String="Path", 
                                color::Symbol=:crimson,
                                linestyle::Symbol = :solid
                                ) 

    # Extraemos coordenadas (Igual que en tu código original)
    xs_path = eval_points[:, 1]
    ys_path = eval_points[:, 2]

    if !isempty(eval_points)
        plot!(plt, xs_path, ys_path, 
            label=label, 
            linecolor=color,
            linestyle = linestyle,
            linewidth=1.5, 
            marker=:circle, 
            markercolor=color,
            markersize=1, 
            alpha=0.7
        )
        # Nota: label=nothing para no ensuciar la leyenda
        scatter!(plt, [xs_path[1]], [ys_path[1]], 
            markercolor=:green, 
            markerstrokecolor=:black,
            marker=:circle, 
            markersize=6, 
            label=nothing 
        )
        # 3. Graficar Fin (mismo color del camino o Rojo/Estrella distintivo)
        scatter!(plt, [xs_path[end]], [ys_path[end]], 
            markercolor=:red, 
            markerstrokecolor=:black,
            marker=:star5, 
            markersize=9, 
            label=nothing
        )
    end
    return plt
end

function plotEvolution(list_of_Historials, problem::String, colors, labels, 
                            styleLine, ylabel::String, plotTitle::String, image_name::String)
    """Function to plot the evolution of a phenomena. 
    
    # Input:
        - list_of_Historials: Array[Vetors] - Array of several historials of the phenomena
        - problem: String - Name of the phenomena
        - colors: Array - Array of colors for the historials (e. g. :blue, :red)
        - labels: Array - Array of labels of each historial 
        - styline: Array - Array of lineStyles to plot the historial (e. g; :dash, :solid)
        - ylabel: String - Name of the Y axis
        - plotTitle: String - Title of the plot
        - image_name: String - Path where to save the plot

    # Output:
        - None: Void - Only creates the plot and save it
        
    # Remark. There is not a check if the length of the historials is the same as for the colors
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
            linewidth = 2
        )
    end
    return plt
end