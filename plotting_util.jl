using LaTeXStrings
using Plots; pythonplot()
using LinearAlgebra

gr()

function plot_optimization_path(f::Function, eval_points::AbstractMatrix; 
                                function_name::String = "Function ", 
                                padding_ratio::Float64 = 0.2, 
                                levels::Int = 30, 
                                resolution::Int = 100,
                                label = nothing,
                                g_xlims, g_ylims)

    # --- 1. CONFIGURACIÓN DEL LIENZO (CONTOUR) ---
    # 1. Definimos lógica de expansión en una mini-función (Calcula límites nuevos + grid)
    #    Nota: max(..., 1.0) evita errores si todos los puntos son iguales (rango 0)
    setup_axis(l) = let d = max(l[2]-l[1], 1.0) * padding_ratio, new_l = (l[1]-d, l[2]+d)
        new_l, range(new_l..., length=resolution)
    end

    # 2. Aplicamos la función a ambos ejes simultáneamente con broadcasting (.)
    #    Esto desempaqueta el resultado en variables limpias
    (xlims, x_grid), (ylims, y_grid) = setup_axis.([g_xlims, g_ylims])

    # 3. Calculamos Z (Matriz de comprensión estándar)
    Z = [f([x, y]) for y in y_grid, x in x_grid]

    # Niveles logarítmicos o lineales
    min_z, max_z = extrema(Z)
    if min_z < 0
        lvl_vals = range(min_z, max_z, length=levels)
    else
        start_val = max(min_z, 1e-5)
        lvl_vals = 10 .^ range(log10(start_val), log10(max_z), length=levels)
    end

    # Crear el objeto Plot inicial (p)
    p = contour(x_grid, y_grid, Z, 
        levels=lvl_vals, color=:plasma, fill=false, alpha=0.4, 
        clabels=false, cbar=true, label=nothing,
        title="$(function_name) Landscape", xlabel="x", ylabel="y"
    )

    # --- 2. DIBUJAR EL PRIMER CAMINO ---
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
                                linestyle::Symbol = :solid) 

    # Extraemos coordenadas (Igual que en tu código original)
    xs_path = eval_points[:, 1]
    ys_path = eval_points[:, 2]

    if !isempty(eval_points)
        # 1. Graficar el camino (Línea)
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

        # 2. Graficar Inicio (Verde)
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

function plotGradsEvolution(list_of_Historials, problem::String, colors, labels, styleLine, image_name::String)
    
    # CORRECCIÓN 1: Usar length() en lugar de len()
    n_elemnts = length(list_of_Historials)

    # Configuración base del lienzo
    plt = plot(
        xlabel="Iteration",
        ylabel="‖∇f(x)‖",
        title="Convergence Analysis: " * problem,
        yscale=:log10,    # Escala logarítmica suele ser vital para gradientes
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3,
        legend=:topright,
        size=(800, 400),
        dpi=300
        # Eliminé 'color=:blue' aquí porque se sobrescribe en el bucle
    )

    for i in 1:n_elemnts
        # Graficamos iterativamente sobre 'plt'
        plot!(plt, list_of_Historials[i],
            label = labels[i],
            linestyle = styleLine[i],
            color = colors[i],
            linewidth = 2  # Un poco más grueso para que se vea el estilo
        )
    end
    
    # Guardar
    savefig(plt, image_name)
end