using GLMakie

struct ConvexPolygon2D
    A::SparseMatrixCSC{Float64, Int64}
    b::Vector{Float64}
    V::Vector{Vector{Float64}}
    function ConvexPolygon2D(A,b,V)
        new(A,b,V)
    end
    function ConvexPolygon2D(A,b; tol=1e-5)
        # Convert from Hrep to Vrep
        m = length(b)
        V = Vector{Float64}[]
        for i in 1:m
            nm = norm(A[i,:])
            A[i,:] ./= nm
            b[i] /= nm
        end
        for i1 = 1:m
            a1 = A[i1,:]
            for i2 = i1+1:m
                a2 = A[i2,:]
                rhs = [b[i1], b[i2]]
                try
                    v = [a1'; a2'] \ -rhs
                    if all(- tol .≤ A*v+b)
                        push!(V, v)
                    end
                catch
                end
            end
        end
        n_verts = length(V)
        c = sum(V) ./ n_verts
    
        θs = map(V) do vi
            d = vi-c
            atan(d[2],d[1])
        end
        I = sortperm(θs) |> reverse
        V = V[I]

        new(A,b,V)
    end
    function ConvexPolygon2D(V; tol=1e-5)
        # Convert from Vrep to Hrep

        A = []
        b = Float64[]
        N = length(V)
        supporting_verts = Int[]
        for i = 1:N
            for j = 1:N
                j == i && continue
                v1 = V[i]
                v2 = V[j]
                t = v2-v1
                t ./= norm(t)
                ai = [t[2], -t[1]]
                bi = -(ai'*v1)
                if all( -tol .≤ ai'*v+bi for v in V)
                    push!(A,ai)
                    push!(b,bi)
                    push!(supporting_verts, i)
                    push!(supporting_verts, j)
                end
            end
        end
        
        A = hcat(A...)' 
        supporting_verts = Set(supporting_verts) |> collect
        V = V[supporting_verts]
        c = sum(V) / length(V)
        θs = map(V) do vi
            d = vi-c
            atan(d[2],d[1])
        end
        I = sortperm(θs) |> reverse
        V = V[I]
        new(sparse(A),b,V)
    end
end

function plot_polys(polys)
    fig = Figure()
    ax = Axis(fig[1,1], aspect=DataAspect())

    colors = [:red, :orange, :yellow, :green]
    for (P,c) in zip(polys,colors)
        plot!(ax, P; color=c)
    end
    display(fig)
end

function gen_polys(N; side_count=4)

    polys = map(1:N) do i
		rng = Random.MersenneTwister(i);
        P = ConvexPolygon2D([randn(rng, 2) + [0,0] for _ in 1:side_count]);
        while length(P.V) != side_count
            P = ConvexPolygon2D([randn(rng, 2) + [0,0] for _ in 1:side_count]);
        end
        P
    end
end

function GLMakie.plot!(ax, P::ConvexPolygon2D; kwargs...)
    N = length(P.V) 
    V = P.V
    for i in 1:N-1
        vii = V[i+1]
        vi = V[i]
        lines!(ax, [vi[1], vii[1]], [vi[2], vii[2]]; kwargs...)
    end
    vii = V[1]
    vi = V[N]
    lines!(ax, [vi[1], vii[1]], [vi[2], vii[2]]; kwargs...)
end
