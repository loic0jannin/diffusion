import pandas as pd


def get_imbalance(df, t, depth):
    # Filter the DataFrame based on the time step
    t_df = df[df['etimestamp'] == t]

    # Calculate the sum of buy volumes up to the given depth
    buy_volumes = t_df[[f'bq{i}' for i in range(depth)]].sum().sum()

    # Calculate the sum of all volumes up to the given depth
    total_volumes = t_df[[f'bq{i}' for i in range(depth)] + [f'aq{i}' for i in range(depth)]].sum().sum()

    # Calculate and return the imbalance
    imbalance = buy_volumes / total_volumes if total_volumes else 0
    return imbalance

def get_absolute_volue(df,t,depth):
    # Filter the DataFrame based on the time step
    t_df = df[df['etimestamp'] == t]

    # Calculate the sum of all volumes up to the given depth
    total_volumes = t_df[[f'bq{i}' for i in range(depth)] + [f'aq{i}' for i in range(depth)]].sum().sum()

    return total_volumes

def get_order_sign_imbalance(df, t, N):
    # Filter the DataFrame based on the time step and select the last N events
    t_df = df[df['etimestamp'] <= t].tail(N)

    # Assign 1 for sell orders and -1 for buy orders
    t_df['order_sign'] = t_df['eside'].apply(lambda x: 1 if x == 'S' else -1)

    # Calculate and return the order-sign imbalance
    return t_df['order_sign'].mean()

def get_spread(df, t):
    # Filter the DataFrame based on the time step
    t_df = df[df['etimestamp'] == t]

    print(t_df['ap0'].item())

    # Calculate the spread
    spread = t_df['ap0'].item() - t_df['bp0'].item()

    return spread

def get_midprice(df, t):
    # Filter the DataFrame based on the time step
    t_df = df[df['etimestamp'] == t]

    # Calculate the mid-price
    mid_price = (t_df['ap0'].item() + t_df['bp0'].item()) / 2

    return mid_price

def get_price_return(df, t, depth):
    # Filter the DataFrame based on the time step and select the last N events
    df = df[df['etimestamp'] <= t].tail(depth)

    # Select the 'etimestamp' value for the first row
    t = df.iloc[0]['etimestamp']

    # Select the 'etimestamp' value for the last row
    t_N = df.iloc[-1]['etimestamp']

    # compute the mid-price at time t and t-N
    mid_price_t = get_midprice(df, t)
    mid_price_t_N = get_midprice(df,t_N)

    # Calculate the price return
    price_return = (mid_price_t/ mid_price_t_N) - 1 

    return price_return

def get_state(data, t):
    # gets the data at time t and the 50 rows before:
    df = data[data['etimestamp'] <= t].tail(50)

    # Calculate the features
    I_1 = get_imbalance(df,t,1)
    I_5 = get_imbalance(df,t,5)
    O_128 = get_order_sign_imbalance(df,t,128)
    O_256 = get_order_sign_imbalance(df,t,256)
    V_1 = get_absolute_volue(df,t,1)
    V_5 = get_absolute_volue(df,t,5)
    S = get_spread(df,t)
    R_1 = get_price_return(df,t,1)
    R_50 = get_price_return(df,t,50)

    # Return the state 
    return [I_1,I_5,O_128,O_256,V_1,V_5,S,R_1,R_50]

def get_action(data, t):
    # gets the event that occured at timestamp t, 


    # initialize the action returned
    action = (
        0,  # depth
        0,  # cancel depth
        0,  # quantity
        0,  # order type
        0  # side
    )

    # Filter the DataFrame for rows where the 'etimestamp' value is equal to the timestamp
    event_at_t = data[data['etimestamp'] == t]

    # Get the order book at the previous timestamp
    event_at_t_minus_1 = data[data['etimestamp'] == t - 1].tail(1)

    # Get the action details
    etype = event_at_t['etype'].values[0]
    eside = 1 if event_at_t['eside'].values[0] == 'B' else -1  # 1 for BUY, -1 for SELL
    eprice = event_at_t['eprice'].values[0]
    esize_ini = event_at_t['esize_ini'].values[0]

    # Determine order type
    if etype == 'A': # add order 
        order_type = 0

        # buy or sell:
        eside = eside

        # Get the depth of the order:
        if eside == 1: # its a buy order
            depth = event_at_t['bp0'].item() - eprice 
        else :
            depth = eprice - event_at_t['ap0'].item() 

        # quantity:
        quantity = esize_ini

    elif etype == 'C': # cancel order
        order_type = -1  

        # buy or sell:
        eside = eside

        # cancel depth :
        if eside == 1:
            cancel_depth = event_at_t['bp0'].item() - eprice
        else:
            cancel_depth = eprice - event_at_t['ap0'].item()
        
        # quantity:
        quantity = esize_ini
        

    if etype == 'M':
        order_type = 1

        # Sort the DataFrame in descending order by 'etimestamp'
        data_sorted = data.sort_values('etimestamp', ascending=False)

        # Search through the DataFrame for the last action involving this order before the timestamp t
        prev_order_index = data_sorted[(data_sorted['eoid'] == eoid) & (data_sorted['etimestamp'] < t)].first_valid_index()

        # If a previous order was found
        if prev_order_index is not None:
            # Get the previous order
            prev_order = data_sorted.loc[prev_order_index]

            # Get the previous state of the order
            prev_eprice = prev_order['eprice'].values[0]
            prev_esize_ini = prev_order['esize_ini'].values[0]
            prev_eside = 1 if prev_order['eside'].values[0] == 'B' else -1

            # computes previous depth:
            if prev_eside == 1:
                prev_depth = prev_order['bp0'].item() - prev_eprice
            else:
                prev_depth = prev_eprice - prev_order['ap0'].item()

            # Get the new state of the order
            eprice = event_at_t['eprice'].values[0]
            new_depth = event_at_t['bp0'].item() - eprice if eside == 1 else eprice - event_at_t['ap0'].item()
            new_quantity = event_at_t['esize_ini'].values[0]

            # Calculate the cancel depth and cancel vsize
            cancel_depth = previous_depth
            cancel_vsize = esize_ini - prev_esize_ini

            # Update the action
            action = (
                new_depth,  # depth
                prev_depth,  # cancel depth
                esize_ini,  # quantity
                etype,  # order type
                eside  # side
            )

    elif etype == 'T': # trade order 
        order_type = 2

        # buy or sell:
        eside = eside

        # Get the depth of the order:
        if eside == 1:
            depth = event_at_t['bp0'].item() - eprice
        else:
            depth = eprice - event_at_t['ap0'].item()
        
        # quantity:
        quantity = esize_ini


    elif etype == 'F':
        order_type = 3

    # Determine quantity
    quantity = esize_ini

    # Calculate depth and cancel depth
    # Assuming bp0 and ap0 are the best bid and ask prices respectively
    depth = event_at_t['bp0'].values[0]
    cancel_depth = event_at_t['ap0'].values[0]

    # Return the event details
    return (depth, cancel_depth, quantity, order_type, eside)
