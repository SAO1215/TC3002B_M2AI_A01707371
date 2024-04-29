import splitfolders

splitfolders.ratio("tom_and_jerry",
                   output = "tom_and_jerry_prep",
                   seed = 42, 
                   ratio = (.7, .2, .1),
                   group_prefix = None,
                   move = False)