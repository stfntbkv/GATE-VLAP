(define (problem test)

    (:domain robotic_planning)
    
    (:objects
        multigrain_chips0 - multigrain_chips
        trash_can0 - trash_can
        grapefruit_soda0 - grapefruit_soda
        7up0 - 7up
        table0 - table
        sprite0 - sprite
        human0 - human
        robot0 - robot_profile
        counter2 - counter
        energy_bar0 - energy_bar
        counter1 - counter
        tea0 - tea
        red_bull0 - red_bull
        pepsi0 - pepsi
        coke0 - coke
        apple0 - apple
        lime_soda0 - lime_soda
        jalapeno_chips0 - jalapeno_chips
        sponge0 - sponge
        water0 - water
    )
    
    (:init 
        (on  sponge0 counter1)
        (on  lime_soda0 counter2)
        (on  jalapeno_chips0 counter2)
        (on  red_bull0 table0)
        (on  multigrain_chips0 counter2)
        (on  grapefruit_soda0 counter1)
        (on  pepsi0 table0)
        (on  tea0 counter2)
        (on  energy_bar0 counter1)
        (on  water0 counter2)
        (at  robot0 counter1)
        (on  coke0 counter1)
        (on  sprite0 table0)
        (on  7up0 counter2)
        (on  apple0 counter2)
        (= total-cost 0)
        (= (cost human0) 100)
        (= (cost robot0) 1)
    )
    
    (:goal (and (in  apple0 trash_can0) (inhand  coke0 human0)))
    (:metric minimize (total-cost))
    
)
