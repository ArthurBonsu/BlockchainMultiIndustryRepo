const BCADN = artifacts.require("BCADN");
const ProactiveDefenseMechanism = artifacts.require("ProactiveDefenseMechanism"); 

// Helper function to add delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

module.exports = async function(deployer, network, accounts) {
  try {
    // Deploy BCADN contract 
    await deployer.deploy(BCADN);
    await delay(5000);  // Add a 5-second delay between deployments
    
    // Deploy Proactive Defense Mechanism
    await deployer.deploy(
      ProactiveDefenseMechanism, 
      30,    // anomalyThreshold
      86400  // baselineUpdateInterval (1 day in seconds)
    );
    
    console.log(`Deployment completed for network: ${network}`);
  } catch (error) {
    console.error('Deployment error:', error);
    throw error; 
  }
};

// Etherscan https://sepolia.etherscan.io/tx/0x3326b4c25c1ae0b7d6db118e86c095f30c9caf052430043ed62955b74abc49c6
//https://sepolia.etherscan.io/block/8118449#consensusinfo