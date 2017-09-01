require 'numo/narray'
require 'pp'


def sigm(x)
    1/(1+Numo::NMath.exp(-x))
end
	
def dsigm(x)
    x*(1-x)
end
	
def tanh(x)
    Numo::NMath.tanh(x)
end
	
def dtanh(x)
	(1-(x.square))
end

class NN
    def initialize(nin, nhidden, nout)
	wstd = 0.1
	@w1 = Numo::DFloat.new(nin, nhidden).rand_norm*wstd
	@wiv = Numo::DFloat.zeros(nin, nhidden)
	@b1 = Numo::DFloat.zeros(nhidden)
	@b1v = Numo::DFloat.zeros(nhidden)

        @wz = Numo::DFloat.new(2*nhidden,nhidden).rand_norm*wstd
        @wzv = Numo::DFloat.zeros(2*nhidden,nhidden) # the weight velocit
        @bz = Numo::DFloat.zeros(nhidden)
        @bzv = Numo::DFloat.zeros(nhidden)
        
        @wr = Numo::DFloat.new(2*nhidden,nhidden).rand_norm*wstd
        @wrv = Numo::DFloat.zeros(2*nhidden,nhidden) # the weight velocit
        @br = Numo::DFloat.zeros(nhidden)
        @brv = Numo::DFloat.zeros(nhidden)
              
        @wh = Numo::DFloat.new(2*nhidden,nhidden).rand_norm*wstd
        @whv = Numo::DFloat.zeros(2*nhidden,nhidden) # the weight velocit
        @bh = Numo::DFloat.zeros(nhidden)
        @bhv = Numo::DFloat.zeros(nhidden)
        
        @w2 = Numo::DFloat.new(nhidden,nout).rand_norm*wstd
        @w2v = Numo::DFloat.zeros(nhidden,nout) # the weight velocit
        @b2 = Numo::DFloat.zeros(nout)
        @b2v = Numo::DFloat.zeros(nout)
        
        @nin = nin
        @nout = nout
	@nhidden = nhidden
    end

    # ''' do the feedforward prediction of a piece of data'''   
    def predict(input)
        l_size = input.shape[0]
        az = Numo::DFloat.zeros(l_size,@nhidden)
        ar = Numo::DFloat.zeros(l_size,@nhidden)
        ahhat = Numo::DFloat.zeros(l_size,@nhidden)
	ah = Numo::DFloat.zeros(l_size,@nhidden)
		
        a1 = tanh((input.dot @w1) + @b1)
        pp "a1 is ============"
        pp a1
        pp a1[1,0...a1.shape[1]]
	    
        # (array slice view) http://ruby-numo.github.io/narray/narray/Numo/DFloat.html#[]-instance_method	    
        x = (Numo::DFloat.zeros(@nhidden)).concatenate(a1[1,0...a1.shape[1]])
        az[1,0...az.shape[1]] = sigm((x.dot @wz) + @bz)
        ar[1,0...ar.shape[1]] = sigm((x.dot @wr) + @br)
        ahhat[1,0...ahhat.shape[1]] = tanh((x.dot @wh) + @bh)
	ah[1,0...ah.shape[1]] = az[1,0...az.shape[1]]*ahhat[1,0...ahhat.shape[1]]

        # for i in range(1,l_size):
        (1...l_size).each do |i|
            x = ah[i-1,0...ah.shape[1]].concatenate(a1[i,0...a1.shape[1]])
            az[i,0...az.shape[1]] = sigm((x.dot @wz) + @bz)
            ar[i,0...ar.shape[1]] = sigm((x.dot @wr) + @br)
            x = (ar[i,0...ar.shape[1]]*ah[i-1,0...ah.shape[1]]).concatenate(a1[i,0...a1.shape[1]])
            ahhat[i,0...ahhat.shape[1]] = tanh((x.dot @wh) + @bh)
            ah[i,0...ah.shape[1]] = (1-az[i,0...az.shape[1]])*ah[i-1,0...az.shape[1]] + az[i,0...az.shape[1]]*ahhat[i,0...ahhat.shape[1]]
        end
 
        a2 = tanh((ah.dot @w2) + @b2)
	return a1,az,ar,ahhat,ah,a2
    end

    def compute_gradients(input,labels)
        a1,az,ar,ahhat,ah,a2 = predict(input)
        error = (labels - a2)
        
        l_size = input.shape[0]
        h_size = @nhidden
        dz = Numo::DFloat.zeros(l_size,h_size)
        dr = Numo::DFloat.zeros(l_size,h_size)
        dh = Numo::DFloat.zeros(l_size,h_size)
		d1 = Numo::DFloat.zeros(l_size,h_size)


        # this is ah from the previous timestep
        # getting array at a position in numo/narray is odd. lot of hacks.
	# ahm1 = (Numo::DFloat.zeros(1,h_size).concatenate(ah[:-1,:])
	ahm1 = Numo::DFloat.zeros(1,h_size).concatenate(ah.delete(-1,0)) # using delete to return everything but the last

        d2 = error*dtanh(a2)
        e2 = error.dot @w2.transpose
	dh_next = Numo::DFloat.zeros(1,@nhidden)
        
        # for i in range(l_size-1,-1,-1):
        (l_size-1).downto(0) do |i|
            err = e2[i,0...e2.shape[1]] + dh_next
            dz[i,0...dz.shape[1]] = (err*ahhat[i,0...ahhat.shape[1]] - err*ahm1[i,0...ahm1.shape[1]])*dsigm(az[i,0...az.shape[1]])
	    dh[i,0...dh.shape[1]] = err*az[i,0...az.shape[1]]*dtanh(ahhat[i,0...ahhat.shape[1]])
	    dr[i,0...dr.shape[1]] = (dh[i,0...dh.shape[1]].dot((@wh.delete(h_size..-1,0)).transpose))*ahm1[i,0...ahm1.shape[1]]*dsigm(ar[i,0...ar.shape[1]])

            dh_next = err*(1-az[i,0...az.shape[1]]) + (dh[i,0...dh.shape[1]].dot(@wh.delete(h_size..-1,0).transpose))*ar[i,0...ar.shape[1]] + (dz[i,0...dz.shape[1]].dot(@wz.delete(h_size..-1,0).transpose)) + (dr[i,0...dr.shape[1]].dot(@wr.delete(h_size..-1,0).transpose))
	    d1[i,0...d1.shape[1]] = (dh[i,0...dh.shape[1]].dot(@wh.delete(0...h_size,0).transpose)) + (dz[i,0...dz.shape[1]].dot(@wz.delete(0...h_size,0).transpose)) + (dr[i,0...dr.shape[1]].dot(@wr.delete(0...h_size,0).transpose))
	end
	d1 = d1*dtanh(a1)

        d1 = d1*dtanh(a1)
        # all the deltas are computed, now compute the gradients
        gw2 = 1.0/l_size * (ah.transpose.dot d2)
        gb2 = 1.0/l_size * d2.sum(axis:0)
        x = ahm1.concatenate(a1, axis:1)
        gwz = 1.0/l_size * (x.transpose.dot dz)
        gbz = 1.0/l_size * dz.sum(axis:0)
        gwr = 1.0/l_size * (x.transpose.dot dr)
        gbr = 1.0/l_size * dr.sum(axis:0)
        x = (ar*ahm1).concatenate(a1, axis:1)
        gwh = 1.0/l_size * (x.transpose.dot dh)
        gbh = 1.0/l_size * dh.sum(axis:0)
        gw1 = 1.0/l_size * (input.transpose.dot d1)
        gb1 = 1.0/l_size * d1.sum(axis:0)
        weight_grads = [gw1,gwr,gwz,gwh,gw2]
	bias_grads = [gb1,gbr,gbz,gbh,gb2]

	puts "++++++++++++++++++++++++++"
        return weight_grads, bias_grads
		
    end

end






# TESTING GRU





# data = Numo::NArray[[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1]]
# data = Numo::NArray[["my unit-tests failed."], 
#                     ["tried the program, but it was buggy."],
#                     ["i need a new power supply."],
#                     ["the drive has a 2TB capacity."],
#                     ["unit-tests"],
#                     ["program"],
#                     ["power supply"],
#                     ["drive"]]
# labels = Numo::NArray[[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1]]
# labels = Numo::NArray[["software"],
#                       ["software"],
#                       ["hardware"],
#                       ["hardware"],
#                       ["software"],
#                       ["software"],
#                       ["hardware"],
#                       ["hardware"]]

data = Numo::NArray[[ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
           [ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 ],
           [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 ],           
           [ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
           [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0 ],
           [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ]]

labels = Numo::NArray[[0],[0],[1],[1],[0],[0],[1],[1]]

net = NN.new(21,5,1) # (input dimensions, hidden layers, number of outputs)

act = net.predict(data)
pp act[-1]
puts "-----------------------------"
d = net.compute_gradients(data,labels)
pp d
puts "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
pp net.predict(Numo::NArray[[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ]])
