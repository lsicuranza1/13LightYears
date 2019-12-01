package game.patterns.state;
 
public class Modalita {
 
    private Stato statoModalita;
 
    public Modalita(){
        this.statoModalita = new StatoStart();
    }

	public Stato getStatoModalita() {
		return statoModalita;
	}

	public void setStatoModalita(Stato statoModalita) {
		this.statoModalita = statoModalita;
	}
 
}