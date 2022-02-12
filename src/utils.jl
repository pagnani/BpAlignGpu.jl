const aminoalphabet = Dict('A'=> 1, 'B'=> 21, 'C' =>2, 'D'=> 3, 'E'=> 4, 'F'=> 5, 'G'=> 6, 'H'=> 7, 'I'=> 8, 'J'=> 21,'K'=> 9, 'L'=> 10, 'M'=> 11, 'N'=>12, 'O'=>21, 'P'=> 13, 'Q'=>14, 'R'=>15, 'S'=>16, 'T'=>17, 'U'=>21, 'V'=>18,'W'=>19, 'X'=>21, 'Y'=>20, 'Z'=> 21, '-'=> 21)
const nbasealphabet = Dict('A'=> 1, 'U'=> 2, 'C'=> 3, 'G'=> 4, '-'=> 5 ,'T'=> 2, 'N'=> 5, 'R'=> 5, 'X' => 5, 'V'=> 5, 'H'=>5, 'D'=>5, 'B'=>5, 'M'=>5, 'W'=>5, 'S'=>5, 'Y'=>5, 'K'=>5)

function letter2num(c::Union{Char,UInt8}, ctype::Symbol)
    if ctype == :amino
        return aminoalphabet[c]
    elseif  ctype == :nbase
        return nbasealphabet[c]
    else
        error("only :amino and :nbase defined")
    end
end