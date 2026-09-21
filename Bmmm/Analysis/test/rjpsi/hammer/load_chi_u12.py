def fitprint(p,u,chi): 
    ret=0 
    
    for j in range(13):
        ret=ret+p[chi+'_A'+str(j)]*(1-u)**(j)
  
    if chi=='chi0-':ret=ret*(((1+u)**2))
    if chi=='chi0+':ret=ret*(((1-u)**2))
    
    return ret

  







