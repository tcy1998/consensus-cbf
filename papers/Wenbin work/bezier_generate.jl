s = 0:1/1200:1;

function Bezier_function(s, P0, P1, P2, P3)
        return (-s.+1).^3 * P0  .+ 3*(-s.+1).^2 .* s * P1 + 3*(-s.+1).* s.^2 * P2 .+ s.^3 * P3
end


function Bezier_d(s, P0, P1, P2, P3)
        un_big = 3*(-s.+1).^2 * (P1-P0) .+ 6*(-s.+1) .* s * (P2-P1) .+ 3*s.^2 * (P3-P2)
        return un_big/(sqrt(un_big' * un_big)) * 5
end

function Bezier_dd(s, P0, P1, P2, P3)
        un_big = 6*(-s.+1) * (P2-2*P1+P0) .+ 6* s * (P3-2*P2+P1)
        return un_big/(sqrt(un_big' * un_big)) * 2
end





""" Curve 1 """
# g = Bezier_function(s, [0 0], [0 200], [200 300], [190 400])

""" Curve 2 """
gg = Bezier_function(s, [200 0], [100 100], [250 200], [200 400])
gg_d = Bezier_d(s,[200 0], [100 100], [250 200], [200 400])
# gg_dd = Bezier_dd(s,[200 0], [100 100], [250 200], [200 400])
gg = [gg gg_d]
""" Curve 3 """
# ggg = Bezier_function(s, [400 0], [200 200], [300 300], [210 400])
