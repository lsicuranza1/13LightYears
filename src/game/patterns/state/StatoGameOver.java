package game.patterns.state;

public class StatoGameOver implements Stato {
 
    @Override
	public void gestioneStato(Modalita modalita, String stato) {
		// TODO Auto-generated method stub
		if (stato.equals("start"))
			modalita.setStatoModalita(new StatoStart());
	}
 
}
