procedure Test(X: in out Integer) is
   procedure F is
   begin
      X := 12;
   end F;
begin
   F;
end Test;
