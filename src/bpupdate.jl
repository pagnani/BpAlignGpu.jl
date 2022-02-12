function update_f!(av::AllVar)
    @extract av : lrf bel
    @extract lrf : f
    @extract bel : conditional

end